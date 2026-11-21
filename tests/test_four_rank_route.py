"""Four genuine CPU/Gloo replicas vs an independent, single-process loss oracle."""
from collections import Counter
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import random
import sys
import unittest

import torch
from torch import distributed as dist, multiprocessing as mp, nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from genevolve_repro import distributed as state
from genevolve_repro.route_sft import RouteScorer, _rank0_checked, update
from genevolve_repro.route_type_kl import prepare_reference_cache

MARK = 15
TYPES = (2, 3, 4)
WORLD = 4
CLIP = (100.0, .22, .30)
CONF = dict(good_per_batch=6, bad_per_batch=2, route_lambda=.1, good_type_kl_lambda=.65)


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(19, 7)
        self.calls = []

    def forward(self, input_ids, **unused):
        self.calls.append(tuple(input_ids[0].tolist()))
        values = self.embedding(input_ids).cumsum(dim=1) * .37
        return SimpleNamespace(last_hidden_state=values)


class Policy(nn.Module):
    def __init__(self, seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.model = Backbone()
            self.lm_head = nn.Linear(7, 19)


class TracedScorer(RouteScorer):
    def __init__(self):
        super().__init__(Policy(543), projection_chunk=2, type_ids=TYPES)
        self.visits = []

    def forward(self, ids, start, is_bad, *args, **kwargs):
        self.visits.append((tuple(ids[0, :start].tolist()), kwargs.get('branch', 'suffix')))
        return super().forward(ids, start, is_bad, *args, **kwargs)


def fixture():
    rows, batches = [], []
    for s, bad_positions in enumerate(({0, 4}, {2, 6}, {5, 7})):
        batch = []
        for j in range(8):
            i = s*8+j
            pref = [8+s, 5+j] + [11+j%3]*(i%3)
            bad = int(j in bad_positions)
            action = ([2] if i%2 else [3, 4]) + [6+i%7]*(i%4) + [16]
            rows.append(dict(id=f'toy-{i}', route_label=bad, target_start=len(pref),
                             input_ids=pref + ([MARK] if bad else []) + action))
            batch.append(i)
        batches.append(batch)
    return rows, batches


def optimizer(net):
    opt = torch.optim.AdamW(net.parameters(), lr=.013, betas=(.86,.93), weight_decay=.04, foreach=False)
    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1/(s+1))


def oracle_three(net, prefix):
    hidden = net.model(torch.tensor([prefix+[3]])).last_hidden_state[0]
    first, second = net.lm_head(hidden[-2:]).softmax(-1)
    tool, answer = first[2], first[3]*second[4]
    return torch.stack((tool, answer, 1-tool-answer))


def snapshot(net, opt):
    # Match the same parameter order on the bare policy and the DDP scorer.
    return [[p.detach().clone(), p.grad.detach().clone(),
             opt.state[p]['exp_avg'].clone(), opt.state[p]['exp_avg_sq'].clone(),
             opt.state[p]['step'].clone()] for p in net.parameters()]


def oracle(rows, batches):
    # Does not call the production scorer, reduction, KL, BCE or teacher cache.
    net, parent = Policy(543), Policy(919)
    opt, sched = optimizer(net)
    with torch.no_grad():
        refs = {i: oracle_three(parent, r['input_ids'][:r['target_start']])
                for i,r in enumerate(rows) if not r['route_label']}
    trace = []
    for s,bb in enumerate(batches):
        opt.zero_grad(set_to_none=True)
        ce, bce, kl = [], [], []
        probs = {0:[], 1:[]}
        for i in bb:
            r = rows[i]
            a, ids, y = r['target_start'], r['input_ids'], r['route_label']
            logits = net.lm_head(net.model(torch.tensor([ids])).last_hidden_state[0])
            ce.append(F.cross_entropy(logits[a-1:-1], torch.tensor(ids[a:])))
            p = logits[a-1].softmax(0)[MARK]
            probs[y].append(p)
            bce.append(-p.log() if y else -torch.log1p(-p))
            if not y:
                q = refs[i]
                kl.append((q * (q.log()-oracle_three(net, ids[:a]).log())).sum())
        mean = lambda t: torch.stack(t).mean()
        ce, bce, kl = mean(ce), mean(bce), mean(kl)
        loss = ce + CONF['route_lambda']*bce + CONF['good_type_kl_lambda']*kl
        loss.backward()
        norm = nn.utils.clip_grad_norm_(net.parameters(), CLIP[s])
        opt.step()
        sched.step()
        metrics = dict(loss=loss, suffix_ce=ce, route_bce=bce, good_p_rethink=mean(probs[0]),
                       bad_p_rethink=mean(probs[1]), good_threeway_kl=kl,
                       weighted_kl=CONF['good_type_kl_lambda']*kl, grad_norm=norm)
        trace.append(dict(metrics={k:float(v.detach()) for k,v in metrics.items()},
                          params=snapshot(net,opt), lr=opt.param_groups[0]['lr']))
    return {i:q.log() for i,q in refs.items()},trace


def worker(rank, temp):
    torch.set_num_threads(1)
    device = torch.device('cpu')
    folder = Path(temp)
    store = dist.FileStore(str(folder/'rendezvous'), WORLD)
    dist.init_process_group('gloo', rank=rank, world_size=WORLD, store=store, timeout=timedelta(seconds=90))
    try:
        def invalid_parent():
            assert rank == 0
            raise ValueError('invalid parent fixture')
        try:
            _rank0_checked(rank, WORLD, invalid_parent)
        except ValueError as exc:
            assert 'invalid parent fixture' in str(exc)
        else:
            raise AssertionError('A failed shared preflight must fail on every rank.')
        rows, batches = fixture()
        ref, trace = torch.load(folder/'oracle.pt', weights_only=True)
        parent = Policy(919)
        contract = {'test':'distributed-route-sft'}
        cache_args = (folder, folder/'parent', rows, contract, rank, WORLD, device)
        cached = prepare_reference_cache(*cache_args, loader=lambda *a, **k: parent,
                                         attention='eager', flush_every=99, type_ids=TYPES)
        wanted = [tuple(rows[i]['input_ids'][:rows[i]['target_start']]+[3])
                  for i in sorted(ref) if i%WORLD == rank]
        assert parent.model.calls == wanted, (rank, 'teacher cache visited the wrong contexts')
        assert cached.keys() == ref.keys()
        for i, value in cached.items():
            torch.testing.assert_close(torch.tensor(value),ref[i], atol=2e-6,rtol=2e-6)
        def must_not_load(*a, **k):
            raise AssertionError('a sealed teacher shard was recomputed')
        assert prepare_reference_cache(*cache_args, loader=must_not_load, attention='eager',
                                       flush_every=99, type_ids=TYPES) == cached
        ddp = DDP(TracedScorer(), broadcast_buffers=False)
        opt, sched = optimizer(ddp)
        for s, bb in enumerate(batches):
            metrics = update(ddp, opt, rows, bb, dict(CONF,clip_grad=CLIP[s]), device, cached,
                             marker_id=MARK)
            sched.step()
            want = Counter()
            for i in bb[rank::WORLD]:
                r = rows[i]
                pref = tuple(r['input_ids'][:r['target_start']])
                want[pref,'suffix'] += 1
                if not r['route_label']:
                    want[pref,'kl'] += 1
            assert Counter(ddp.module.visits) == want, (rank, s, 'cases or branches repeated or missing')
            ddp.module.visits.clear()
            exp = trace[s]
            assert metrics.keys() == exp['metrics'].keys()
            for k in metrics:
                torch.testing.assert_close(metrics[k], exp['metrics'][k], atol=4e-6, rtol=2e-5,
                                           msg=f'rank {rank} update {s} metric {k}')
            assert opt.param_groups[0]['lr'] == exp['lr']
            for got, expected in zip(snapshot(ddp,opt),exp['params'],strict=True):
                for x,y in zip(got,expected,strict=True):
                    torch.testing.assert_close(x,y,atol=3e-6,rtol=3e-5)
            if s==1:
                random.seed(200+rank)
                torch.manual_seed(200+rank)
                state.save_checkpoint(folder/'resume', ddp,opt,sched,2,contract,rank,device,verify=True)
                next_random = (random.random(), torch.rand(3))
                new = DDP(TracedScorer(),broadcast_buffers=False)
                opt, sched = optimizer(new)
                chk = state.load_checkpoint(folder/'resume'/'checkpoint-00002')
                assert chk['step']==2 and len(chk['rng'])==WORLD and chk['contract']==contract
                new.module.load_state_dict(chk['actor'])
                state.restore_optimizer(new,opt,sched,chk,rank)
                state.restore_rng(chk['rng'][rank],device)
                assert random.random() == next_random[0]
                assert torch.equal(torch.rand(3),next_random[1])
                ddp = new
    finally:
        dist.destroy_process_group()


class TestFourRankRoute(unittest.TestCase):
    def test_real_gloo_exact_gradients_updates_shards_and_resume(self):
        self.assertTrue(dist.is_gloo_available(), 'The required four-rank test needs CPU Gloo.')
        if sys.platform == 'win32':
            # Some Windows CPU wheels expose Gloo but ship without a transport.
            # Never skip on Linux (CI), nor silence genuine rendezvous/test failures.
            try:
                dist.ProcessGroupGloo.create_device(hostname='127.0.0.1')
            except RuntimeError as exc:
                if 'unsupported gloo device' not in str(exc).lower():
                    raise
                self.skipTest('This Windows PyTorch build has no Gloo transport; use Linux or a Gloo-capable wheel.')
        with TemporaryDirectory() as tmp:
            prev = torch.get_num_threads()
            try:
                torch.set_num_threads(1)
                result = oracle(*fixture())
            finally:
                torch.set_num_threads(prev)
            self.assertTrue(all(item['metrics']['good_threeway_kl'] > 0 for item in result[1]))
            self.assertLess(result[1][0]['metrics']['grad_norm'],CLIP[0])
            self.assertGreater(result[1][1]['metrics']['grad_norm'],CLIP[1])
            torch.save(result, Path(tmp)/'oracle.pt')
            mp.spawn(worker, args=(tmp,), nprocs=WORLD, join=True)


if __name__ == '__main__':
    unittest.main()
