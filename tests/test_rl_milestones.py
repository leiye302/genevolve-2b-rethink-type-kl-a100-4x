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

        def train():
            order.append(("launch", 40))
            for step in plan.STOPS:
                done.add(step)
                if step < 40:
                    score(plan.policy_path(work, step), 8)

        def score(policy, maximum):
            reports.add((policy, maximum))
            order.append((policy.name, maximum))

        with patch.object(plan, "require_restartable"), \
                patch.object(plan, "verified_policy", side_effect=lambda w, s, step: step in done), \
                patch.object(evaluate, "bench_complete", side_effect=lambda w, p, m: (p, m) in reports):
            plan.run_schedule(work, {}, train, score)
            first = list(order)
            plan.run_schedule(work, {}, train, score)
        self.assertEqual(order, first)
        self.assertEqual(first, [("launch",40), ("hf-step-00010",8), ("hf-step-00020",8),
                                 ("hf-step-00030",8), ("final",8), ("final",None)])
        self.assertNotEqual(evaluate.eval_folder(work, work / "rethink/final", "bench", 8),
                            evaluate.eval_folder(work, work / "rethink/final", "bench", None))

    def test_partial_weights_cannot_be_used_for_mid_rl_resume(self):
        with TemporaryDirectory() as d:
            work=Path(d)
            plan.require_restartable(work)
            write_json(plan.policy_path(work,10)/"completed.json",{"step":10})
            with self.assertRaisesRegex(ValueError,"fresh WORK"):
                plan.run_schedule(work, {}, self.fail, self.fail)

    def test_live_fixed8_child_isolated_from_worker_and_waits_on_cpu(self):
        work,policy=Path("work"),Path("work/rethink/hf-step-00020")
        group = object()
        with patch.dict(os.environ, {"RANK":"0","WORLD_SIZE":"4","MASTER_PORT":"12",
                                     "CUDA_VISIBLE_DEVICES":"3,5,7,2","IMAGE_DOWNLOAD_DIR":"training-cache"}), \
                patch.object(plan.torch.cuda,"empty_cache"), \
                patch.object(plan.torch.cuda,"mem_get_info",return_value=(60*2**30,80*2**30)), \
                patch.object(evaluate,"bench_complete",side_effect=[False,True]), \
                patch.object(plan.subprocess,"run") as launch, \
                patch.object(plan.dist,"broadcast_object_list") as wait:
            plan.live_fixed8(work,policy,0,group)
            args=launch.call_args
            self.assertEqual(args.args[0][1:],["-m","genevolve_repro.evaluate","--work",str(work),"--policy",str(policy)])
            env=args.kwargs["env"]
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"],"3")
            self.assertFalse({"MASTER_PORT","WORLD_SIZE","RANK"} & set(env))
            self.assertEqual(env["IMAGE_DOWNLOAD_DIR"],str(work/"cache/eval-rank0"))
            self.assertEqual(os.environ["IMAGE_DOWNLOAD_DIR"],"training-cache")
            wait.assert_called_once_with([""],src=0,group=group)

    def test_corrupt_checkpoint_invalidates_handoff_even_if_export_metadata_intact(self):
        with TemporaryDirectory() as d:
            work = Path(d)
            policy = plan.policy_path(work, 40)
            cp = work / "rethink/checkpoint-00040"
            cp.mkdir(parents=True)
            ct = {"algorithm": plan.ALGORITHM, "total_steps": 40, "world_size": 4,
                  "sft": {"origin": "fixture"}, "milestones": list(plan.STOPS), "storage": plan.STORAGE}
            write_json(work / "rethink/contract.json", ct)
            model = cp / "training.pt"
            model.write_bytes(b"valid")
            sha = file_sha256(model)
            write_json(cp / "state_sha256.json", {"training.pt": sha})
            write_json(policy / "completed.json", {"step": 40, **plan.export_meta(ct, sha)})
            write_json(policy / "export_sha256.json", {"fixture": True})
            with patch.object(plan, "sft_manifest", return_value={"fixture": True}):
                self.assertTrue(plan.verified_policy(work, ct["sft"], 40))
                model.write_bytes(b"wrong")
                with self.assertRaises(ValueError):
                    plan.verified_policy(work, ct["sft"], 40)

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
            stack.enter_context(patch.object(evolve,"control_group",return_value=object()))
            seen = []
            def probe(work,policy,rank,group):
                n=read_json(policy/"completed.json")["step"]
                self.assertEqual(read_json(policy/"completed.json")["artifact"],"weights_only")
                self.assertEqual(list((work/"rethink").glob("checkpoint-*")),[])
                self.assertFalse(list((work/"rethink").rglob("training.pt")))
                self.assertTrue((work/"rethink/metrics"/f"step-{n:05d}.json").is_file())
                self.assertFalse((work/"rethink/metrics"/f"step-{n+1:05d}.json").exists())
                # Even deliberate consumption IN the trainer is isolated by its
                # in-memory save/restore, in addition to actual subprocess isolation.
                random.seed(987+n); torch.manual_seed(654+n); torch.rand(1)
                seen.append(n)
            stack.enter_context(patch.object(evolve,"live_fixed8",probe))
            evolve._loop(base, root/"sft", 0, 4, device, 40, rethink=True)
            with patch.object(evolve,"save_checkpoint",wraps=evolve.save_checkpoint) as save:
                evolve._loop(resumed,root/"sft",0,4,device,40,rethink=True,milestone_eval=True)
                self.assertEqual([c.args[4] for c in save.call_args_list],[40])
            self.assertEqual(seen,[10,20,30])
            # Crash AFTER the sole native save but BEFORE finishing the final HF export.
            marker = plan.policy_path(resumed,40) / "completed.json"
            marker.unlink()
            marker.parent.rmdir()
            evolve._loop(resumed,root/"sft",0,4,device,40,rethink=True,milestone_eval=True)
            self.assertEqual(read_json(marker)["step"],40)
            self.assertEqual(seen,[10,20,30])
            left, right = [load_checkpoint(w / "rethink/checkpoint-00040") for w in (base, resumed)]
            self.assertEqual([p.name for p in (resumed/"rethink").glob("checkpoint-*")], ["checkpoint-00040"])
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
