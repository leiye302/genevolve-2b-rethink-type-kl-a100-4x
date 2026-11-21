import json
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

import run
from genevolve_repro import hints, teacher
from genevolve_repro.common import ROOT, read_json, write_json
from genevolve_repro.distributed import load_checkpoint, save_checkpoint
from genevolve_repro import evolve
from genevolve_repro.evolve import update_actor
from genevolve_repro.evaluate import evaluate_case
from genevolve_repro.memory import ExperienceBank
from genevolve_repro.objectives import advantages, choose_distillation_tokens, loss_terms
from genevolve_repro.policy import Turn, Trajectory, teacher_prefix
from genevolve_repro.route_data import MARKER_ID, recipe, source_checkpoint_lock, verify_parent
from genevolve_repro.route_sft import RouteScorer, verify_route_export
from genevolve_repro.sft_export import sft_manifest
from test_contract import fake_export


class ToyTokenizer:
    all_special_ids = [MARKER_ID]
    eos_token_id = 2

    def apply_chat_template(self, messages, **kwargs):
        return list("\n".join(x["content"] for x in messages).encode())

    def encode(self, text, **kwargs):
        return list(text.encode())


class ToyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ToyBackbone()
        self.lm_head = torch.nn.Linear(5, 13)


class ToyBackbone(torch.nn.Embedding):
    def __init__(self):
        super().__init__(13, 5)

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(last_hidden_state=super().forward(input_ids).cumsum(dim=1))


class RethinkTest(unittest.TestCase):
    def fixture(self):
        messages = [{"role": "system", "content": "native instructions\n# Tools\ncallable tools"},
                    {"role": "user", "content": "Draw a gold bird."}]
        tagged = '<think>Check the evidence.</think>\n<RETHINK><tool_call>{"name":"image_search"}</tool_call>'
        normal = '<think>Use the image.</think>\n<answer>{"gen_prompt":"Sketch the bird"}</answer>'
        turn1 = Turn([1, 2], [4, MARKER_ID, 2, 3], old=torch.zeros(4),
                     messages=messages, response=tagged, action=[2, 3])
        turn2_messages = messages + [{"role": "assistant", "content": tagged},
                                     {"role": "user", "content": "<tool_response>\nIMG_001\n"
                                      "  local_path: /app/cache/private.png\nhttps://example.test/a?sig=example"
                                      "\n</tool_response>"}]
        turn2 = Turn([1, 3], [1, 8], old=torch.zeros(2),
                     messages=turn2_messages, response=normal, action=[1])
        other = Turn([1], [1, 8], old=torch.zeros(2), messages=messages, response=normal, action=[1])
        result = SimpleNamespace(
            messages=turn2_messages + [{"role": "assistant", "content": normal}],
            termination="answer", gt_image="should not be shared")
        traj = Trajectory(result, [turn1, turn2])
        group = SimpleNamespace(trajectories=[traj, Trajectory(result, [other])])
        return group, traj, turn1, turn2, other

    def native_cfg(self):
        return {**read_json(ROOT / "configs" / "evolve.json"),
                **read_json(ROOT / "configs" / "genevolve_sdl.json")}

    def test_route_mean_ce_and_full_vocabulary_bce_use_the_predictive_index(self):
        torch.manual_seed(0)
        model = ToyPolicy()
        scorer = RouteScorer(model, projection_chunk=2)
        # The first trainable target is marker 7 at offset 3. Earlier IDs
        # condition the suffix but must not contribute additional CE terms.
        ids = torch.tensor([[4, 1, 9, 7, 8, 10, 2]])
        lam = recipe()["route_lambda"]
        loss, ce, bce, p = scorer(ids, 3, 1, 7, lam)
        logits = model.lm_head(model.model(ids).last_hidden_state[0])
        expected_ce = F.cross_entropy(logits[2:-1], ids[0, 3:])
        expected_p = logits[2].softmax(0)[7]
        expected = expected_ce - lam * expected_p.log()
        torch.testing.assert_close(loss, expected)
        torch.testing.assert_close(ce, expected_ce.detach())
        torch.testing.assert_close(bce, -expected_p.detach().log())
        torch.testing.assert_close(p, expected_p.detach())
        expected_grad = torch.autograd.grad(expected, model.lm_head.weight)[0]
        torch.testing.assert_close(torch.autograd.grad(loss, model.lm_head.weight)[0], expected_grad)

    def test_only_generated_id_enables_local_teacher_hint(self):
        group, traj, first, later, unrelated = self.fixture()
        original = [(list(t.prefix), list(t.tokens), [m.copy() for m in t.messages])
                    for t in (first, later, unrelated)]
        scored = []

        def client(settings, cfg, jobs):
            self.assertEqual(jobs, [(traj, [1])])
            # Even if the provider attempts to suggest an unselected step, it
            # has no route and cannot add the suggestion to a Teacher row.
            return [({1: "Use the image result.", 2: "INJECTION"}, 1)]

        def fake_score(_, prefix, tokens, temperature, device):
            self.assertFalse(torch.is_grad_enabled())
            scored.append((bytes(prefix).decode(), tokens))
            return torch.arange(len(tokens), dtype=torch.float32)

        cfg = self.native_cfg()
        tok = ToyTokenizer()
        guidance = '{"decision_focus":"native prior experience"}'
        with patch("genevolve_repro.teacher.score_turn", side_effect=fake_score):
            stats = teacher.score_groups([group], None, tok, cfg, "cpu", None, [guidance], client)
        self.assertEqual((stats["turns"], stats["generated_marker"], stats["eligible"],
                          stats["api_attempts"], stats["local_used"], stats["teacher_turns"]), (3, 1, 1, 1, 1, 3))
        self.assertEqual(stats["native_boundary"], 1)
        self.assertTrue(all(f"<visual_experience>\n{guidance}\n</visual_experience>" in prefix
                            for prefix, _ in scored))
        self.assertIn("Use the image result.", scored[0][0])
        self.assertEqual([prefix for prefix, _ in scored[1:]],
                         [bytes(teacher_prefix(tok, t.messages, guidance)).decode() for t in (later, unrelated)])
        self.assertTrue(all("INJECTION" not in prefix for prefix, _ in scored))
        self.assertEqual([ids for _, ids in scored], [t.tokens for t in (first, later, unrelated)])
        self.assertEqual(original, [(t.prefix, t.tokens, t.messages) for t in (first, later, unrelated)])
        self.assertTrue(all(not t.teacher.requires_grad for t in (first, later, unrelated)))
        self.assertEqual(first.distill.nonzero().flatten().tolist(), [3])
        self.assertFalse(later.distill.any())
        self.assertEqual(stats["sdl_tokens"], 2)

    def test_no_api_for_unmarked_and_malformed_marker_is_not_a_second_gate(self):
        group, trajectory, first, _, _ = self.fixture()
        first.response = first.response.replace("<RETHINK>", "", 1) + "<RETHINK>"
        self.assertFalse(teacher.is_routed(first))

        def client(settings, cfg, jobs):
            self.assertEqual(jobs, [(trajectory, [1])])
            return [({}, 1)]

        _, stats = teacher.plan_hints([group], teacher.recipe(), None, client)
        self.assertEqual((stats["generated_marker"], stats["native_boundary"], stats["eligible"]), (1, 0, 1))
        first.tokens.remove(MARKER_ID)
        with patch("genevolve_repro.hints.request_trajectories") as api:
            incoming, stats = teacher.plan_hints([group], teacher.recipe(), None)
            api.assert_not_called()
        self.assertFalse(incoming)
        self.assertEqual(stats["generated_marker"], 0)
        self.assertEqual(stats["eligible"], 0)

    def test_missing_or_ungrounded_hint_falls_back(self):
        group, traj, first, _, _ = self.fixture()
        tok = ToyTokenizer()
        guidance = "original experience"
        expected = teacher_prefix(tok, first.messages, guidance)
        with patch("genevolve_repro.teacher.score_turn", side_effect=lambda _, p, ids, *a: torch.zeros(len(ids))) as score:
            stats = teacher.score_groups([group], None, tok, self.native_cfg(),
                                         "cpu", None, [guidance], lambda *_: [({}, 2)])
        self.assertEqual((stats["eligible"], stats["native_fallback"], stats["local_used"]), (1, 1, 0))
        self.assertEqual(score.call_args_list[0].args[1], expected)
        for invalid in ("x" * 97, "Use IMG_001."):
            self.assertEqual(teacher.preferred_prefix(tok, first, guidance, invalid, 65536, 96), (expected, False))
        self.assertEqual(teacher.preferred_prefix(tok, first, guidance, "Check the wing.",
                                                  len(expected) + len(first.tokens), 96), (expected, False))
        _, has_local = teacher.preferred_prefix(tok, traj.turns[1], guidance, "Use IMG_001.", 65536, 96)
        self.assertTrue(has_local)
        self.assertEqual(teacher.preferred_prefix(tok, first, guidance, None, 3, 96), (None, False))
        self.assertEqual(score.call_count, 3)

    def test_empty_bank_does_not_create_a_static_teacher(self):
        group, _, first, later, unrelated = self.fixture()
        with patch("genevolve_repro.teacher.score_turn", return_value=torch.zeros(len(first.tokens))) as score:
            stats = teacher.score_groups([group], None, ToyTokenizer(), self.native_cfg(),
                                         "cpu", None, [None], lambda *_: [({1: "Search for the wing."}, 1)])
        self.assertEqual((stats["local_used"], stats["teacher_turns"]), (1, 1))
        self.assertNotIn(b"<visual_experience>", bytes(score.call_args.args[1]))
        self.assertIsNone(later.teacher)
        self.assertIsNone(unrelated.teacher)

    def test_api_request_contains_only_redacted_trajectory_and_parses_selected_steps(self):
        _, trajectory, _, _, _ = self.fixture()
        body = hints.build_payload(trajectory, [1])
        dumped = json.dumps(body)
        self.assertNotIn("private.png", dumped)
        self.assertNotIn("sig=", dumped)
        self.assertNotIn("should not be shared", dumped)
        self.assertEqual(body["selected_turns"], [1])
        self.assertEqual([m["turn"] for m in body["timeline"]], [0, 1, 1, 2])
        content = '{"hints":{"1":"Search for the wing pattern.","2":"not selected"}}'
        reply = {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}
        self.assertEqual(hints.parse_reply(reply, [1]), {1: "Search for the wing pattern."})
        reply["choices"][0]["message"]["content"] = '{"hints":{"1":"a","1":"b"}}'
        with self.assertRaises(ValueError):
            hints.parse_reply(reply, [1])

    def test_eval_marker_telemetry_cannot_change_rendering_score(self):
        _, traj, first, _, _ = self.fixture()
        traj.result.gen_prompt = "Sketch the bird"
        traj.result.reference_images = []
        traj.result.to_dict = lambda: {}
        example = SimpleNamespace(id=1, gt=None, prompt="Draw a gold bird")
        feedback = SimpleNamespace(renderer=object(), score_image=lambda *_: {"overall": 0.64})
        with patch("genevolve_repro.evaluate.render_for_evaluation", return_value={"success": True, "render_fallback": True}):
            marked = evaluate_case(SimpleNamespace(collect=lambda _: traj), feedback, example, {}, Path("fixture.png"))
            first.tokens.remove(MARKER_ID)
            plain = evaluate_case(SimpleNamespace(collect=lambda _: traj), feedback, example, {}, Path("fixture.png"))
        self.assertEqual(marked.pop("route_telemetry"),
                         {"turns": 2, "marker_turns": 1, "native_boundary_turns": 1})
        self.assertEqual(plain.pop("route_telemetry"),
                         {"turns": 2, "marker_turns": 0, "native_boundary_turns": 0})
        self.assertEqual(marked, plain)

    def test_native_sdl_only_top_ten_percent_action_tokens_and_detached_teacher(self):
        cfg = self.native_cfg()
        target = torch.tensor([-20.0, -1.0, -3.0, -5.0], requires_grad=True)
        turn = Turn([1], [1, MARKER_ID, 2, 3], old=torch.tensor([-2.0, -2.0, -2.0, -2.0]),
                    teacher=target.detach(), action=[2, 3])
        self.assertEqual(choose_distillation_tokens([turn], cfg["sdl_top_fraction"]), 1)
        self.assertEqual(turn.distill.tolist(), [False, False, False, True])
        turn.teacher = target
        student = torch.tensor([-2.2, -2.0, -1.6, -1.7], requires_grad=True)
        _, sdl = loss_terms(student, turn, 0.6, cfg)
        d = student[-1] - target[-1].detach()
        ratio = (student[-1] - turn.old[-1]).exp().clamp(max=2.0)
        expected = (ratio * (-d).expm1().add(d).clamp(0, 10)).clamp(max=10)
        torch.testing.assert_close(sdl, expected)
        sdl.backward()
        self.assertIsNone(target.grad)
        self.assertEqual(student.grad[:3].tolist(), [0.0] * 3)
        self.assertNotEqual(student.grad[-1].item(), 0.0)

    def test_mixed_turn_update_keeps_all_grpo_tokens(self):
        torch.manual_seed(0)
        model = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        a = Turn([1], [1, 1], old=torch.zeros(2), reference=torch.zeros(2),
                 teacher=torch.tensor([-0.2, -1.0]), distill=torch.tensor([False, True]))
        b = Turn([1], [1, 1, 1], old=torch.zeros(3), reference=torch.zeros(3))
        group = SimpleNamespace(trajectories=[SimpleNamespace(turns=[a]), SimpleNamespace(turns=[b])],
                                outcomes=[SimpleNamespace(reward=1), SimpleNamespace(reward=0)])
        cfg = self.native_cfg()

        def scorer(actor, prefix, tokens, temperature, device):
            return actor(torch.ones(len(tokens), 1)).flatten() * 0

        with patch("genevolve_repro.evolve.score_turn", side_effect=scorer) as score:
            grpo, norm, sdl = update_actor(model, optimizer, [group], cfg, torch.device("cpu"), 2,
                                          use_sdl=True)
            self.assertEqual(score.call_count, 2)
        positive, negative = advantages([1, 0])
        self.assertAlmostEqual(grpo, -(2 * positive + 3 * negative) / 2)
        self.assertAlmostEqual(sdl, (torch.exp(torch.tensor(-1.0)).item()) / 2)
        self.assertEqual(norm, 0.0)

    def test_native_experience_bank_survives_checkpoint(self):
        memory = ExperienceBank(500)
        bundle = {"decision_guidance": {"decision_focus": "check the wing pattern"}}
        memory.update([{"vector": [0.0, 1.0], "gap": 0.5, "source_id": "fixture", "bundle": bundle}])
        model = ToyPolicy()
        optimizer = torch.optim.AdamW(model.parameters())
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 2)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_checkpoint(root, model, optimizer, scheduler, 1, {"algorithm": "gen-evolve"}, 0,
                            torch.device("cpu"), verify=True, memory=memory)
            state = load_checkpoint(root / "checkpoint-00001")
            self.assertEqual(state["memory"], memory.entries)
            self.assertEqual(ExperienceBank(500, state["memory"]).retrieve([0.0, 1.0]),
                             json.dumps(bundle["decision_guidance"]))

    def test_two_stage2_updates_use_native_experience_only_after_writing_it(self):
        device = torch.device("cpu")
        tok = ToyTokenizer()
        example = SimpleNamespace(id="fixture", prompt="Draw a gold bird.")
        actor = torch.nn.Linear(1, 1, bias=False)
        bundle = {"decision_guidance": {"decision_focus": "native experience fixture"}}
        feedback = SimpleNamespace(summarize=lambda *args: bundle)
        teacher_inputs = []

        def rollout(agent, feedback, example, path, count):
            self.assertEqual(count, 6)
            group, _, _, _, _ = self.fixture()
            outcomes = [SimpleNamespace(reward=r, valid=True) for r in (0.9, 0.1)]
            return evolve.Group(example, group.trajectories, outcomes)

        def token_scores(scorer, prefix, tokens, temperature, device):
            return scorer(torch.ones(len(tokens), 1)).flatten()

        def teacher_scores(scorer, prefix, tokens, temperature, device):
            teacher_inputs.append(bytes(prefix).decode())
            return token_scores(scorer, prefix, tokens, temperature, device) - 0.25

        def request(settings, cfg, jobs):
            self.assertTrue(jobs)
            self.assertTrue(all(selected == [1] for _, selected in jobs))
            return [({1: "Check the wing shape."}, 1) for _ in jobs]

        with TemporaryDirectory() as temp, patch.multiple(
            evolve, load_upstream=lambda: None, required_env=lambda _: "offline-fixture",
            embedding_ready=lambda _: True, examples=lambda *_: [example] * 2575, verify_split=lambda *_: "fixture-sha",
            verify_route_export=lambda _: {"completed": 562}, make_sharded_actor=lambda *args: actor,
            load_model=lambda *args: deepcopy(actor), ActorScorer=lambda model: model,
            copy_actor=lambda src, dst: dst.load_state_dict(src.state_dict()), Embedder=lambda *_: lambda _: [1.0],
            LiveFeedback=lambda *args: feedback, make_agent=lambda *a, **k: None, gather_group=rollout,
            score_turn=token_scores, write_rollouts=lambda *a, **k: None, export_policy=lambda *args: None,
        ), patch("transformers.AutoTokenizer.from_pretrained", return_value=tok), \
                patch("transformers.AutoProcessor.from_pretrained"), \
                patch("genevolve_repro.teacher.verify_marker"), \
                patch("genevolve_repro.teacher.score_turn", side_effect=teacher_scores), \
                patch("genevolve_repro.hints.settings_from_env", return_value=SimpleNamespace(identity=lambda: {})), \
                patch("genevolve_repro.hints.request_trajectories", side_effect=request):
            root = Path(temp)
            evolve._loop(root, root / "route", 0, 4, device, 2, rethink=True)
            saved = load_checkpoint(root / "rethink" / "checkpoint-00002")
            metrics1 = read_json(root / "rethink" / "metrics" / "step-00001.json")
            metrics2 = read_json(root / "rethink" / "metrics" / "step-00002.json")
            self.assertEqual((metrics1["experience_entries"], metrics2["experience_entries"]), (2, 4))
            self.assertEqual((metrics1["sdl_tokens"], metrics2["sdl_tokens"]), (2, 4))
            self.assertEqual(len(saved["memory"]), 4)
            self.assertNotIn("sdar_coef", saved["contract"]["config"])
            self.assertTrue(all("<visual_experience>" not in text for text in teacher_inputs[:2]))
            self.assertTrue(all("<visual_experience>" in text for text in teacher_inputs[2:]))
            self.assertEqual(sum("<local_decision_hint>" in text for text in teacher_inputs), 4)

    def test_stage2_clamps_to_exact_complete_route_export(self):
        cfg = recipe()
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "route"
            parent = Path(tmp) / "parent"
            fake_export(parent)
            fake_export(path)
            with self.assertRaisesRegex(ValueError, "configured"):
                verify_route_export(path)
            identity = {
                "config": cfg, "manifest_sha256": cfg["manifest_sha256"],
                "route_data_source_sha256": source_checkpoint_lock(),
                "parent": read_json(ROOT / "configs/sft550_sha256.json"),
                "artifact_sha256": {k+".jsonl": read_json(ROOT / "configs/route_original_v4.json")[k+"_sha256"]
                                   for k in ("train","batches")},
            }
            write_json(path / "completed.json", {"step": 562, "algorithm": "route-sft", "contract": identity})
            write_json(path / "export_sha256.json", sft_manifest(path))
            self.assertEqual(verify_route_export(path, expected_contract=identity)["completed"]["step"], 562)
            (parent / "model.safetensors").write_bytes(b"a different step 550 export")
            with self.assertRaisesRegex(ValueError, "supplied 2B parent"):
                verify_route_export(path, expected_contract={**identity, "parent": sft_manifest(parent)})
            (path / "model.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                verify_route_export(path)


if __name__ == "__main__":
    unittest.main()
