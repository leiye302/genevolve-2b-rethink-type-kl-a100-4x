"""Linux four-rank FSDP with mixed labels, checkpointing and global Good mean."""
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import torch
import torch.distributed as dist

from genevolve_repro.distributed import copy_actor
from genevolve_repro.route_sft import RouteScorer, update
from genevolve_repro.route_type_kl import VERSION
from test_distributed import sharded
from test_pipeline import tiny_policy


def rank_worker(rank, directory):
    torch.set_num_threads(1)
    folder = Path(directory)
    device = torch.device('cpu')
    dist.init_process_group('gloo', init_method=(folder / 'rendezvous').as_uri(), rank=rank,
                            world_size=4, timeout=timedelta(seconds=120))
    try:
        torch.manual_seed(42)
        policy = tiny_policy()
        policy.set_attn_implementation('sdpa')
        policy.model.visual.requires_grad_(False)
        baseline = deepcopy(policy)
        policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        actor = sharded(RouteScorer(policy, type_ids=(6, 3, 4)), device)
        reference = RouteScorer(baseline, type_ids=(6, 3, 4))
        # Mixed ranks: 0,1 have one Good and one Bad; 2,3 have two Good.
        rows = [dict(input_ids=[2, 5, 9, 6, 8, 2], target_start=3, route_label=int(i in (0, 5)))
                for i in range(8)]
        refs = {i: [float(x) for x in torch.tensor([0.3, 0.7]).log()] for i in range(8) if i not in (0, 5)}
        cfg = dict(micro_batch=1, good_per_batch=6, bad_per_batch=2, route_lambda=0.1,
                   clip_grad=1.0, good_type_kl_lambda=0.1, good_type_kl_version=VERSION)
        optimizer = torch.optim.SGD(actor.parameters(), lr=0.02)
        expected_optimizer = torch.optim.SGD(reference.parameters(), lr=0.02)
        # route marker is intentionally within the tiny model's vocabulary.
        import genevolve_repro.route_sft as module
        module.MARKER_ID = 7
        for iteration in range(2):
            expected_optimizer.zero_grad(set_to_none=True)
            for i, row in enumerate(rows):
                ids = torch.tensor([row['input_ids']])
                logp = torch.tensor(refs.get(i, [-0.69314718056] * 2))
                loss = reference(ids, 3, row['route_label'], 7, 0.1, 0.1 * 8 / 6, logp)[0]
                (loss / 8).backward()
            expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
            expected_optimizer.step()
            stats, norm = update(actor, optimizer, rows, list(range(8)), cfg, rank, 4, device, refs)
            assert len(stats) == 2
            torch.testing.assert_close(torch.tensor(norm), expected_norm, atol=3e-5, rtol=3e-5)
            actual = tiny_policy()
            copy_actor(actor, actual)
            for a, b in zip(actual.parameters(), baseline.parameters()):
                torch.testing.assert_close(a, b, atol=3e-5, rtol=3e-5)
    finally:
        dist.destroy_process_group()


class DistributedTypeKLTest(unittest.TestCase):
    @unittest.skipUnless(sys.platform == 'linux' and dist.is_gloo_available(), 'Linux Gloo required')
    def test_four_ranks_two_updates_match_global_reference(self):
        with TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(rank_worker, args=(directory,), nprocs=4)
