"""Exercise the public orchestrator without GPUs, checkpoints or paid services."""
from argparse import Namespace
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, call, patch

import run


class ThreeStages(unittest.TestCase):
    def setUp(self):
        self.work = Path(self.enterContext(TemporaryDirectory())).resolve()
        self.args = Namespace(command="all", parent=None, data=None, steps=40, eval=False)
        self.native = self.work / "native_sft" / "checkpoint-550"
        self.route = run.final_route_export(self.work)
        self.pack = self.work / "route-data"
        stack = self.enterContext(ExitStack())
        self.events = Mock()
        for name, target in {
            "native": "genevolve_repro.native_sft.run_native",
            "pack": "genevolve_repro.route_release.prepare_route",
            "upstream": "run.load_upstream", "services": "run.check_rethink_services",
            "env": "run.required_env", "hf": "run.export_files", "launch": "run.distributed",
            "verify_data": "genevolve_repro.data.verify_data", "prepare": "genevolve_repro.data.prepare",
            "prepare_embedding": "genevolve_repro.data.prepare_embedding",
            "embedding_ready": "genevolve_repro.data.embedding_ready",
            "manifest": "genevolve_repro.route_data.pack_manifest",
            "parent": "genevolve_repro.route_data.verify_parent",
            "identity": "genevolve_repro.route_data.route_identity",
            "hardware": "genevolve_repro.runtime.preflight",
            "space": "genevolve_repro.runtime.check_milestone_space",
            "verified": "genevolve_repro.route_sft.verify_route_export",
            "bench": "genevolve_repro.rl_milestones.run_bench_eval",
            "schedule": "genevolve_repro.rl_milestones.run_schedule",
        }.items():
            self.events.attach_mock(stack.enter_context(patch(target)), name)
        self.events.native.return_value = self.native
        self.events.pack.return_value = self.pack
        self.events.embedding_ready.return_value = False
        self.events.bench.side_effect = lambda work,policy,count,fn: fn(policy,count)
        # Drive the exact public callback interface. The independent milestone tests
        # additionally exercise scoring contracts, partial results and real RNG restoration.
        def schedule(work, identity, train, evaluate):
            self.assertIs(identity, self.events.verified.return_value)
            for n in (10,20,30,40):
                train(n)
                policy = work / "rethink" / ("final" if n == 40 else f"hf-step-{n:05d}")
                evaluate(policy, 8)
            evaluate(policy, None)
        self.events.schedule.side_effect = schedule

    def evalcall(self, policy, n=8):
        return call("_eval", self.work, "--split", "bench", "--policy", str(policy),
                    *(("--max-cases", str(n)) if n is not None else ("--full-benchmark",)))

    def test_default_three_stages_dense_schedule_no_midpoint_or_extra_full_sft_eval(self):
        self.events.verify_data.side_effect = FileNotFoundError()
        run.execute(self.args, self.work)
        self.events.prepare.assert_called_once_with(self.work)
        self.events.native.assert_called_once_with(self.work)
        self.events.parent.assert_called_once()
        self.events.prepare_embedding.assert_called_once()
        expected = [self.evalcall(self.native),
                    call("_route-sft", self.work, "--parent", str(self.native), "--data", str(self.pack)),
                    self.evalcall(self.route)]
        for n in (10, 20, 30, 40):
            expected += [call("_rethink", self.work, "--sft-checkpoint", str(self.route),
                              "--steps", "40", "--stop-after", str(n)),
                         self.evalcall(self.work / "rethink" / ("final" if n == 40 else f"hf-step-{n:05d}"))]
        expected += [self.evalcall(self.work / "rethink" / "final", None)]
        self.assertEqual(self.events.launch.call_args_list, expected)
        self.events.verified.assert_called_once_with(self.route, expected_contract=self.events.identity.return_value)

    def test_explicit_existing_parent_and_verified_route_skip_training_but_not_eval_gates(self):
        self.args.parent = self.native
        self.route.mkdir(parents=True)
        for f in ("completed.json","export_sha256.json"):
            (self.route / f).touch()
        run.execute(self.args, self.work)
        self.events.native.assert_not_called()
        self.assertNotIn("_route-sft", [c.args[0] for c in self.events.launch.call_args_list])
        self.assertEqual([c.args[1:3] for c in self.events.bench.call_args_list], [(self.native,8),(self.route,8)])
        self.events.parent.assert_called_once()

    def test_no_unverified_parent_or_wrong_route_is_evaluated(self):
        self.events.parent.side_effect = ValueError("wrong parent")
        with self.assertRaisesRegex(ValueError,"wrong parent"):
            run.execute(self.args,self.work)
        self.events.bench.assert_not_called()
        self.events.parent.side_effect = None
        self.events.verified.side_effect = ValueError("wrong Route export")
        with self.assertRaisesRegex(ValueError,"wrong Route export"):
            run.execute(self.args,self.work)
        self.assertEqual(self.events.bench.call_args_list[-1].args[1], self.native)
        self.events.schedule.assert_not_called()

    def test_eval_error_stops_before_next_training(self):
        self.events.bench.side_effect = RuntimeError("judge offline")
        with self.assertRaisesRegex(RuntimeError,"judge offline"):
            run.execute(self.args,self.work)
        self.events.schedule.assert_not_called()
        self.events.launch.assert_not_called()

    def test_invalid_horizon_rejected_before_external_calls(self):
        self.args.steps = 322
        with self.assertRaisesRegex(ValueError,"40"):
            run.execute(self.args,self.work)
        self.events.services.assert_not_called()
        self.events.native.assert_not_called()

    def test_route_sft_alone_needs_no_services_and_evals_only_if_requested(self):
        args = Namespace(command="route-sft", parent=self.native, data=self.pack, eval=False)
        train_call = call("_route-sft", self.work, "--parent", str(self.native), "--data", str(self.pack))
        run.execute(args,self.work)
        self.assertEqual(self.events.launch.call_args_list, [train_call])
        self.events.services.assert_not_called()
        self.events.env.assert_not_called()
        args.eval = True
        self.events.launch.reset_mock()
        run.execute(args,self.work)
        self.assertEqual(self.events.launch.call_args_list, [train_call, self.evalcall(self.route)])
        self.events.verified.assert_called_once_with(self.route)


if __name__ == "__main__":
    unittest.main()
