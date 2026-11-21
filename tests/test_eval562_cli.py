"""Hardware-only orchestration and fixed-eight evaluation contract (CPU only)."""
import argparse
from pathlib import Path
import unittest
from unittest.mock import patch

import run
from genevolve_repro.common import ROOT, read_json


class TableContract(unittest.TestCase):
    def test_default_explicit_full_and_custom_are_distinct(self):
        parser = argparse.ArgumentParser()
        run.eval_flags(parser)
        for argv, expected in [([], 8), (["--max-cases", "3"], 3), (["--full-benchmark"], None)]:
            args = parser.parse_args(argv)
            self.assertEqual(args.max_cases, expected)
            internal = parser.parse_args(run.evaluation_arguments(args))
            self.assertEqual(internal.max_cases, expected)

    def test_both_training_stages_use_four_workers_but_evaluation_keeps_one(self):
        with patch.object(run.subprocess, "run") as command:
            for stage, workers in [("_route-sft", 4), ("_eval", 1), ("_rethink", 4), ("_grpo", 4)]:
                run.distributed(stage, Path("work"))
                self.assertIn(f"--nproc-per-node={workers}", command.call_args.args[0])

    def test_exact_eight_sequence_and_decoding(self):
        cfg = read_json(ROOT / "configs" / "evaluation.json")
        self.assertEqual(cfg["selected_case_ids"], [str(i) for i in range(8)])
        self.assertEqual((cfg["full_count"],cfg["max_cases"],cfg["seed"],cfg["temperature"],cfg["top_p"]),
                         (594,8,42,.6,.9))
        self.assertEqual(read_json(ROOT/"configs"/"split_sha256.json")["GenEvolve-Bench/test.jsonl"],
                         "2ff3d33785dc8c0d72ae72d42a05088faa8e392d5c09e6998588605f757d4c90")

    def test_fixed8_actual_turn_denominators_and_infrastructure_failures(self):
        from genevolve_repro.evaluate import fixed_eight_metrics
        rows = [{"id": str(i), "success": True, "eval_success": True, "image_path": "fixture",
                 "scores": {"overall": .8}, "route_telemetry": {"marker_turns": int(i == 0),
                 "native_boundary_turns": int(i == 0), "turns": 4}} for i in range(8)]
        rows[-1].update(success=False, eval_success=False, render_fallback=True,
                        image_status="requires at least one valid reference image")
        result = fixed_eight_metrics(rows, image_exists=lambda _: True)
        self.assertAlmostEqual(result["score_all8_missing_zero"], .7)
        self.assertEqual((result["images_scored"], result["marker_turns"], result["all_turns"],
                          result["marker_turn_rate"], result["marker_case_rate"]), (7, 1, 32, 1/32, 1/8))
        rows[-1]["image_status"] = "upstream renderer timed out"
        with self.assertRaisesRegex(ValueError, "service failure"):
            fixed_eight_metrics(rows, image_exists=lambda _: True)


if __name__ == "__main__":
    unittest.main()
