"""CPU-only train/resume/evaluation contracts, without model downloads or APIs."""
from contextlib import ExitStack, redirect_stdout
from io import StringIO
from pathlib import Path
import random
import os
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from genevolve_repro import evolve, evaluate, rl_milestones as plan
from genevolve_repro.common import read_json, write_json
from genevolve_repro.distributed import load_checkpoint
from genevolve_repro.sft_export import file_sha256


class MilestoneTest(unittest.TestCase):
    def test_exact_cadence_two_scopes_and_no_repeat_on_reentry(self):
        work = Path("work")
        done = set()
        reports = set()
        order = []

        def train(step):
            done.add(step)
            order.append(("train", step))

        def score(policy, maximum):
            reports.add((policy, maximum))
            order.append((policy.name, maximum))

        with patch.object(plan, "verified_policy", side_effect=lambda w, s, step: step in done), \
                patch.object(evaluate, "bench_complete", side_effect=lambda w, p, m: (p, m) in reports):
            plan.run_schedule(work, {}, train, score)
            first = list(order)
            plan.run_schedule(work, {}, train, score)
        self.assertEqual(order, first)
        self.assertEqual(first, [("train", 10), ("hf-step-00010", 8), ("train", 20), ("hf-step-00020", 8),
                                 ("train", 30), ("hf-step-00030", 8), ("train", 40), ("final", 8), ("final", None)])
        self.assertNotEqual(evaluate.eval_folder(work, work / "rethink/final", "bench", 8),
                            evaluate.eval_folder(work, work / "rethink/final", "bench", None))
        for parameters in [(10, 10, True), (40, 15, True), (40, 10, False)]:
            with self.assertRaises(ValueError):
                plan.stop_step(*parameters)

    def test_corrupt_checkpoint_invalidates_handoff_even_if_export_metadata_intact(self):
        with TemporaryDirectory() as d:
            work = Path(d)
            policy = plan.policy_path(work, 10)
            cp = work / "rethink/checkpoint-00010"
            cp.mkdir(parents=True)
            ct = {"algorithm": plan.ALGORITHM, "total_steps": 40, "world_size": 4,
                  "sft": {"origin": "fixture"}, "milestones": list(plan.STOPS)}
            write_json(work / "rethink/contract.json", ct)
            model = cp / "training.pt"
            model.write_bytes(b"valid")
            sha = file_sha256(model)
            write_json(cp / "state_sha256.json", {"training.pt": sha})
            write_json(policy / "completed.json", {"step": 10, **plan.export_meta(ct, sha)})
            write_json(policy / "export_sha256.json", {"fixture": True})
            with patch.object(plan, "sft_manifest", return_value={"fixture": True}):
                self.assertTrue(plan.verified_policy(work, ct["sft"], 10))
                model.write_bytes(b"wrong")
                with self.assertRaises(ValueError):
                    plan.verified_policy(work, ct["sft"], 10)

    def test_594_cases_accept_official_missing_image_not_transport_errors(self):
        expected = list(map(str, range(594)))
        rows = [{"id": k, "route_telemetry": {"turns": 2, "marker_turns": 0, "native_boundary_turns": 0},
                 "success": True, "eval_success": True, "scores": {"overall": .5}, "image_path": "image"}
                for k in expected]
        rows[520].update(success=False, eval_success=False, render_fallback=True,
                         image_status="requires at least one valid reference image")
        result = evaluate.bench_metrics(rows, expected, image_exists=lambda p: True)
        self.assertEqual(result["cases"], 594)
        self.assertEqual(result["images_scored"], 593)
        self.assertAlmostEqual(result["score_missing_zero"], .5 * 593 / 594)
        rows[520]["image_status"] = "backend unavailable"
        with self.assertRaisesRegex(ValueError, "service failure"):
            evaluate.bench_metrics(rows, expected, image_exists=lambda p: True)
        with self.assertRaisesRegex(ValueError, "IDs"):
            evaluate.bench_metrics(rows[:-1], expected)

    def test_cached_bad_case_is_retried_but_valid_case_is_reused(self):
        with TemporaryDirectory() as d:
            folder = Path(d)
            case = folder / "cases/00002.json"
            png = folder / "images/00002.png"
            png.parent.mkdir()
            png.write_bytes(b"good-pixels")
            row = {"id": "2", "route_telemetry": {"turns": 1, "marker_turns": 0, "native_boundary_turns": 0},
                   "success": True, "eval_success": True, "scores": {"overall": .5}, "image_path": str(png)}
            write_json(case, row)
            seal = {"sha256": {"cases/00002.json": file_sha256(case)}, "image_sha256": {"2": file_sha256(png)}}
            self.assertTrue(evaluate.valid_cached_case(case, "2", png, seal))
            png.write_bytes(b"damaged")
            self.assertFalse(evaluate.valid_cached_case(case, "2", png, seal))
            self.assertFalse(evaluate.valid_cached_case(case, "3", png))
            row.update(success=False, eval_success=False, render_fallback=False, image_status="timeout")
            write_json(case, row)
            self.assertFalse(evaluate.valid_cached_case(case, "2", png))

    def test_real_official_tables_receipt_and_selective_regeneration(self):
        # Retain production folder, ledger, score aggregation, receipts and seeds.
        # Substitute ONLY expensive model/tool/Judge and use a synthetic pinned pool.
        with TemporaryDirectory() as d, ExitStack() as stack, redirect_stdout(StringIO()):
            work = Path(d)
            policy = work / "rethink/final"
            pool = [SimpleNamespace(id=str(i), prompt=str(i)) for i in range(594)]
            stack.enter_context(patch.dict(os.environ))
            for name, replacement in {
                "load_upstream": lambda: None, "required_env": lambda _: None, "model_ready": lambda _: True,
                "verify_split": lambda *a: "pinned", "examples": lambda *a: pool,
                "jsonl": lambda *a: ({"id": r.id} for r in pool),
                "sft_manifest": lambda p: {"bytes": "sha256"}, "load_model": lambda *a: torch.nn.Linear(1,1),
                "make_agent": lambda *a: object(), "LiveFeedback": lambda *a: object(),
            }.items():
                stack.enter_context(patch.object(evaluate, name, replacement))
            stack.enter_context(patch("transformers.AutoTokenizer.from_pretrained", return_value=object()))
            seen = []
            def sample(agent, feedback, example, metadata, image):
                seen.append((example.id, torch.initial_seed()))
                image.parent.mkdir(exist_ok=True, parents=True)
                image.write_bytes(b"fixture pixels")
                return {"id": example.id, "route_telemetry": {"turns":2, "marker_turns":0,"native_boundary_turns":0},
                        "success":True, "eval_success":True, "scores":{"overall":.5},
                        "image_path":str(image), "eval_type":"Knowledge-Anchored"}
            stack.enter_context(patch.object(evaluate, "evaluate_case", sample))
            callback = Mock(side_effect=lambda p,m: evaluate._evaluate(work,p,"bench",m,0,1,"cpu"))
            plan.run_bench_eval(work,policy,8,callback)
            plan.run_bench_eval(work,policy,8,callback)
            self.assertEqual(seen, [(str(i),42+i) for i in range(8)])
            self.assertEqual(callback.call_count,1)
            folder = evaluate.eval_folder(work,policy,"bench",8)
            (folder/"images/00005.png").write_bytes(b"wrong pixels")
            plan.run_bench_eval(work,policy,8,callback)
            self.assertEqual(seen[8:], [("5",47)])
            self.assertTrue(evaluate.bench_complete(work,policy,8))
            self.assertFalse(evaluate.bench_complete(work,policy,None))

    def test_real_adam_scheduler_bank_and_rng_match_uninterrupted_40(self):
        # Real optimizer, cosine scheduler, safe on-disk checkpoints, and recovery
        # code; stub actual GPUs, external services, rollouts and multi-rank collectives.
        with TemporaryDirectory() as d, ExitStack() as stack, redirect_stdout(StringIO()):
            root = Path(d)
            base, resumed = root / "continuous", root / "resumed"
            examples = [SimpleNamespace(id=str(i), prompt=str(i)) for i in range(2575)]
            device = torch.device("cpu")
            def actor(parent, dev, saved):
                inst = torch.nn.Linear(2, 1)
                if saved:
                    inst.load_state_dict(saved)
                return inst
            def update(model, opt, *a, **k):
                opt.zero_grad(set_to_none=True)
                loss = model(torch.rand(2) + random.random()).sum()
                loss.backward()
                opt.step()
                return loss.item(), 1., 0.
            def group(agent, feedback, example, path, n):
                return evolve.Group(example, [1] * n, [SimpleNamespace(reward=.4, valid=True)] * n)
            def export(folder, actor, replica, processor, step, rank, metadata=None, **kw):
                write_json(folder / "completed.json", {"step": step, **(metadata or {})})
            def embedder(path):
                # Tests that setup consuming RNG cannot change the resumed trajectory.
                random.random(); torch.rand(1)
                return lambda prompt: [1.]
            def feedback(*a, **k):
                return SimpleNamespace(summarize=lambda *args: {"decision_guidance": "example"})
            actors = {name: value for name, value in {
                "load_upstream": lambda: None, "required_env": lambda x: None, "embedding_ready": lambda w: True,
                "examples": lambda *a: list(examples), "verify_route_export": lambda p: {"sft": "fixture"},
                "verify_split": lambda *a: "split", "gather": lambda x: [x], "make_sharded_actor": actor,
                "load_model": lambda *a: torch.nn.Linear(2,1), "Embedder": embedder, "LiveFeedback": feedback,
                "make_agent": lambda *a, **k: object(), "copy_actor": lambda *a: None, "gather_group": group,
                "score_group": lambda *a: None, "contrast_pair": lambda *a: (1, 2, .4),
                "write_rollouts": lambda *a, **k: None, "score_reference": lambda *a: None, "update_actor": update,
                "export_policy": export}.items()}
            for name, value in actors.items():
                stack.enter_context(patch.object(evolve, name, value))
            stack.enter_context(patch("genevolve_repro.hints.settings_from_env",
                                     return_value=SimpleNamespace(identity=lambda: {})))
            stack.enter_context(patch.object(evolve.teacher, "verify_marker"))
            stack.enter_context(patch.object(evolve.teacher, "score_groups", return_value={}))
            stack.enter_context(patch("transformers.AutoProcessor.from_pretrained", return_value=object()))
            stack.enter_context(patch("transformers.AutoTokenizer.from_pretrained",
                                     return_value=SimpleNamespace(eos_token_id=0)))
            stack.enter_context(patch("genevolve_repro.distributed.gather", side_effect=lambda x: [x] * 4))
            for work, stops in ((base, [None]), (resumed, [10,20,30,40])):
                for stop in stops:
                    evolve._loop(work, root / "sft", 0, 4, device, 40, rethink=True, stop_after=stop)
                    if stop == 10:
                        # Simulate loss of the not-yet-finished HF handoff AFTER native save.
                        export_marker = plan.policy_path(work, stop) / "completed.json"
                        export_marker.unlink()
                        export_marker.parent.rmdir()
                        evolve._loop(work, root / "sft", 0, 4, device, 40, rethink=True, stop_after=stop)
                        self.assertEqual(read_json(export_marker)["step"], 10)
            left, right = [load_checkpoint(w / "rethink/checkpoint-00040") for w in (base, resumed)]
            self.assertEqual(len(list((resumed / "rethink").glob("checkpoint-*"))), 4)
            self.assertEqual(left["scheduler"], right["scheduler"])
            self.assertEqual(right["scheduler"]["T_max"], 40)
            self.assertEqual(left["memory"], right["memory"])
            self.assertGreater(len(right["memory"]), 0)
            self.assertEqual(left["rng"][0]["python"], right["rng"][0]["python"])
            torch.testing.assert_close(left["rng"][0]["torch"], right["rng"][0]["torch"], rtol=0, atol=0)
            for key, value in left["actor"].items():
                torch.testing.assert_close(value, right["actor"][key], rtol=0, atol=0)
            for row, states in left["optimizer"]["state"].items():
                for name, value in states.items():
                    torch.testing.assert_close(value, right["optimizer"]["state"][row][name], rtol=0, atol=0)
