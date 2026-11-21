from argparse import Namespace
from contextlib import ExitStack, redirect_stdout
from io import StringIO
from pathlib import Path
from subprocess import CalledProcessError
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import ANY, Mock, call, patch

import run


class LaunchAllTest(unittest.TestCase):
    def setUp(self):
        directory = self.enterContext(TemporaryDirectory())
        root = Path(directory).resolve()
        self.work = root / "work"
        self.parent = root / "checkpoint-550"
        self.pack = root / "route-sft-v4"
        self.final = self.work / "route_sft" / "hf-step-00562"
        self.args = Namespace(command="all", parent=self.parent, data=self.pack, steps=30, eval=False)
        stack = self.enterContext(ExitStack())
        stack.enter_context(redirect_stdout(StringIO()))
        self.sequence = Mock()
        for name, target in {
            "public_pack": "genevolve_repro.route_release.prepare_route",
            "upstream": "run.load_upstream",
            "services": "run.check_rethink_services",
            "env": "run.required_env",
            "export": "run.export_files",
            "launch": "run.distributed",
            "verify_data": "genevolve_repro.data.verify_data",
            "prepare": "genevolve_repro.data.prepare",
            "prepare_embedding": "genevolve_repro.data.prepare_embedding",
            "embedding_ready": "genevolve_repro.data.embedding_ready",
            "manifest": "genevolve_repro.route_data.pack_manifest",
            "parent": "genevolve_repro.route_data.verify_parent",
            "identity": "genevolve_repro.route_data.route_identity",
            "hardware": "genevolve_repro.runtime.preflight",
            "handoff": "genevolve_repro.route_sft.verify_route_export",
        }.items():
            self.sequence.attach_mock(stack.enter_context(patch(target)), name)
        self.sequence.embedding_ready.return_value = False
        self.sequence.public_pack.return_value = self.pack

    def test_download_then_route_sft_then_verify_then_stage2_then_eval(self):
        self.sequence.verify_data.side_effect = FileNotFoundError()
        self.args.eval = True
        run.execute(self.args, self.work)
        self.sequence.assert_has_calls([
            call.verify_data(self.work), call.prepare(self.work),
            call.launch("_route-sft", self.work, "--parent", str(self.parent), "--data", str(self.pack)),
        ])
        self.sequence.assert_has_calls([
            call.handoff(self.final, expected_contract=self.sequence.identity.return_value),
            call.embedding_ready(self.work), call.prepare_embedding(self.work),
            call.hardware("rethink", self.work),
            call.launch("_rethink", self.work, "--sft-checkpoint", str(self.final), "--steps", "30"),
        ])
        self.assertEqual(self.sequence.launch.call_args_list[-1],
                         call("_eval", self.work, "--split", "bench", "--policy",
                              str(self.work / "rethink" / "final"), "--max-cases", "8"))

    def test_resume_reuses_ready_stage1_and_pinned_data(self):
        self.final.mkdir(parents=True)
        for name in ("completed.json", "export_sha256.json"):
            (self.final / name).touch()
        self.sequence.embedding_ready.return_value = True
        run.execute(self.args, self.work)
        self.sequence.prepare.assert_not_called()
        self.sequence.prepare_embedding.assert_not_called()
        self.sequence.handoff.assert_called_once_with(
            self.final, expected_contract=self.sequence.identity.return_value,
        )
        self.sequence.launch.assert_called_once_with(
            "_rethink", self.work, "--sft-checkpoint", str(self.final), "--steps", "30",
        )
        self.sequence.public_pack.assert_not_called()

    def test_all_uses_public_pack_when_no_override_is_given(self):
        self.args.data = None
        run.execute(self.args, self.work)
        self.sequence.public_pack.assert_called_once_with(self.work)
        self.sequence.manifest.assert_called_once_with(self.pack, ANY)
        self.sequence.launch.assert_any_call("_route-sft", self.work, "--parent", str(self.parent),
                                             "--data", str(self.pack))

    def test_pack_failure_stops_before_training(self):
        self.args.data = None
        self.sequence.public_pack.side_effect = ValueError("dataset checksum mismatch")
        with self.assertRaisesRegex(ValueError, "dataset checksum mismatch"):
            run.execute(self.args, self.work)
        self.sequence.hardware.assert_not_called()
        self.sequence.launch.assert_not_called()

    def test_stage1_exit_failure_never_starts_stage2(self):
        self.sequence.launch.side_effect = CalledProcessError(1, "fixture")
        with self.assertRaises(CalledProcessError):
            run.execute(self.args, self.work)
        self.sequence.launch.assert_called_once()
        self.sequence.handoff.assert_not_called()
        self.sequence.prepare_embedding.assert_not_called()

    def test_route_only_eval_is_called_once_after_final_verified_and_not_at_281(self):
        args = Namespace(command="route-sft", parent=self.parent, data=self.pack, eval=True)
        run.execute(args, self.work)
        self.assertEqual(self.sequence.launch.call_args_list, [
            call("_route-sft", self.work, "--parent", str(self.parent), "--data", str(self.pack)),
            call("_eval", self.work, "--split", "bench", "--policy", str(self.final), "--max-cases", "8"),
        ])
        self.sequence.handoff.assert_called_once_with(self.final)
        self.sequence.services.assert_not_called()

    def test_route_only_plain_command_never_evaluates_or_needs_services(self):
        run.execute(Namespace(command="route-sft", parent=self.parent, data=self.pack, eval=False), self.work)
        self.sequence.launch.assert_called_once_with("_route-sft", self.work, "--parent", str(self.parent),
                                                   "--data", str(self.pack))
        self.sequence.env.assert_not_called()
        self.sequence.prepare.assert_not_called()

    def test_failed_final_handoff_never_pays_for_evaluation(self):
        self.sequence.handoff.side_effect = ValueError("invalid export")
        with self.assertRaisesRegex(ValueError, "invalid export"):
            run.execute(Namespace(command="route-sft", parent=self.parent, data=self.pack, eval=True), self.work)
        self.assertEqual([x.args[0] for x in self.sequence.launch.call_args_list], ["_route-sft"])

    def test_bad_route_export_blocks_stage2(self):
        self.sequence.handoff.side_effect = ValueError("fixture export mismatch")
        with self.assertRaisesRegex(ValueError, "fixture export mismatch"):
            run.execute(self.args, self.work)
        self.assertEqual([record.args[0] for record in self.sequence.launch.call_args_list], ["_route-sft"])

    def test_invalid_steps_fail_before_any_stage(self):
        self.args.steps = 0
        with self.assertRaisesRegex(ValueError, "must be positive"):
            run.execute(self.args, self.work)
        self.sequence.verify_data.assert_not_called()
        self.sequence.launch.assert_not_called()

    def test_wrong_parent_stops_before_downloading_data_or_launching(self):
        self.sequence.parent.side_effect = ValueError("not a 2B SFT export")
        with self.assertRaisesRegex(ValueError, "not a 2B SFT export"):
            run.execute(self.args, self.work)
        self.sequence.verify_data.assert_not_called()
        self.sequence.launch.assert_not_called()
