from copy import deepcopy
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from genevolve_repro.common import read_json, write_json
from genevolve_repro.route_sft import RouteScorer, route_bce
from genevolve_repro.route_type_kl import (
    VERSION, candidate_prefix, prepare_reference_cache, reference_to_student_kl,
    score_reference, type_log_probs, validate_config, verify_type_tokenizer,
)
from test_rethink import ToyPolicy


IDS = (6, 3, 4)


class TypeKLTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_native_prefix_probability_and_first_marker_cancellation(self):
        first = torch.randn(13, requires_grad=True)
        after = torch.randn(13, requires_grad=True)
        logp = type_log_probs(first, after, IDS)
        p = first.softmax(0)
        expected = torch.stack((p[6], p[3] * after.softmax(0)[4]))
        expected = expected / expected.sum()
        torch.testing.assert_close(logp.exp(), expected)
        shifted = type_log_probs(first + 40, after - 80, IDS)
        torch.testing.assert_close(logp, shifted, atol=1e-5, rtol=1e-5)
        reference = torch.tensor([0.3, 0.7]).log().requires_grad_(True)
        kl = reference_to_student_kl(logp, reference)
        expected_kl = (reference.detach().exp() * (reference.detach() - logp)).sum()
        torch.testing.assert_close(kl, expected_kl)
        kl.backward()
        self.assertIsNone(reference.grad)
        self.assertEqual(first.grad[7].item(), 0.0)  # new route marker not a type
        self.assertNotEqual(first.grad[3].item(), 0.0)
        self.assertNotEqual(after.grad[4].item(), 0.0)

    def test_disabled_kl_is_exact_original_loss_and_gradient_with_one_forward(self):
        model = ToyPolicy()
        scorer = RouteScorer(model, projection_chunk=2, type_ids=IDS)
        ids = torch.tensor([[4, 1, 9, 7, 8, 10, 2]])
        with patch.object(model.model, 'forward', wraps=model.model.forward) as forward:
            loss, ce, bce, probability = scorer(ids, 3, 1, 7, 0.1)
            self.assertEqual(forward.call_count, 1)
        # Match the original chunked reduction order, not a single large
        # reduction whose final float32 rounding can differ.
        states = model.model(ids).last_hidden_state[0, 2:-1]
        def ce_sum(x, y):
            return F.cross_entropy(model.lm_head(x).float(), y, reduction='sum')
        original_ce = torch.stack([checkpoint(ce_sum, x, y, use_reentrant=False)
                                   for x, y in zip(states.split(2), ids[0, 3:].split(2))]).sum() / 4
        original_bce, original_p = route_bce(model.lm_head(states[0]), 7, 1)
        original = original_ce + 0.1 * original_bce
        torch.testing.assert_close(loss, original, rtol=0, atol=0)
        torch.testing.assert_close(ce, original_ce.detach())
        torch.testing.assert_close(bce, original_bce.detach())
        torch.testing.assert_close(probability, original_p.detach())
        left = torch.autograd.grad(loss, tuple(model.parameters()), retain_graph=True)
        right = torch.autograd.grad(original, tuple(model.parameters()))
        for x, y in zip(left, right):
            torch.testing.assert_close(x, y, rtol=0, atol=0)

    def test_good_type_kl_is_not_conditioned_on_target_or_marker(self):
        model = ToyPolicy()
        scorer = RouteScorer(model, type_ids=IDS)
        parent = deepcopy(model).eval().requires_grad_(False)
        reference_ids = torch.tensor([[4, 1, 9, 6, 8, 2]])
        reference = score_reference(parent, reference_ids, 3, IDS)
        with torch.no_grad():
            model.lm_head.bias[6] += 0.5
        probabilities = []
        for suffix in ([6, 8, 2], [3, 4, 8, 2]):
            ids = torch.tensor([[4, 1, 9] + suffix])
            with patch.object(model.model, 'forward', wraps=model.model.forward) as forward:
                result = scorer(ids, 3, 0, 7, 0.1, 0.1, reference)
                self.assertEqual(forward.call_count, 2)
                self.assertEqual(forward.call_args.kwargs['input_ids'].tolist(), [[4, 1, 9, 3]])
            self.assertGreater(result[4].item(), 0)
            probabilities.append(result[5])
            result[0].backward()
            model.zero_grad(set_to_none=True)
        torch.testing.assert_close(*probabilities)
        self.assertTrue(all(p.grad is None for p in parent.parameters()))
        self.assertEqual(candidate_prefix(reference_ids, 3, IDS).tolist(), [[4, 1, 9, 3]])

    def test_bad_zero_kl_keeps_two_forwards_and_original_gradients(self):
        first = ToyPolicy()
        second = deepcopy(first)
        ids = torch.tensor([[4, 1, 9, 7, 8, 2]])
        original = RouteScorer(first, type_ids=IDS)(ids, 3, 1, 7, 0.1)[0]
        scorer = RouteScorer(second, type_ids=IDS)
        with patch.object(second.model, 'forward', wraps=second.model.forward) as forward:
            result = scorer(ids, 3, 1, 7, 0.1, 0.1, torch.tensor([-math.log(2)] * 2))
            self.assertEqual(forward.call_count, 2)
        self.assertEqual(result[4].item(), 0)
        original.backward()
        result[0].backward()
        for a, b in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)

    def test_good_mean_not_batch_mean(self):
        values = torch.arange(1, 9, dtype=torch.float32)
        good = torch.tensor([True, True, False, True, True, True, False, True])
        weighted = (values * good * (8 / 6)).mean()
        torch.testing.assert_close(weighted, values[good].mean())

    def test_reference_cache_good_only_seals_resumes_and_rejects_other_runs(self):
        rows = [dict(id=str(i), route_label=i % 2, input_ids=[4, 1, 9, 6, 8, 2], target_start=3)
                for i in range(4)]
        model = ToyPolicy()
        original = deepcopy(model.state_dict())
        calls = []

        def loader(*args):
            calls.append(args)
            return model

        original_score = score_reference

        def score(policy, ids, start):
            self.assertEqual(ids.tolist(), [[4, 1, 9]])
            self.assertFalse(policy.training)
            self.assertTrue(all(not p.requires_grad for p in policy.parameters()))
            return original_score(policy, ids, start, IDS)

        with TemporaryDirectory() as directory, patch('genevolve_repro.route_type_kl.score_reference', side_effect=score):
            root = Path(directory)
            args = (root, root / 'parent', rows, {'test': 1}, 0, 1, torch.device('cpu'))
            cached = prepare_reference_cache(*args, loader=loader)
            self.assertEqual(set(cached), {0, 2})
            self.assertEqual(cached, prepare_reference_cache(*args, loader=loader))
            self.assertEqual(len(calls), 1)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, original[key], rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, 'another parent'):
                prepare_reference_cache(root, root / 'parent', rows, {'test': 2}, 0, 1,
                                        torch.device('cpu'), loader=loader)
            path = root / 'good_type_reference/rank-0.json'
            payload = read_json(path)
            payload['entries']['0']['logp'] = [-1.0, -1.0]
            write_json(path, payload)
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                prepare_reference_cache(*args, loader=loader)

    def test_config_validation_and_native_token_contract(self):
        for value in (-0.1, float('nan'), float('inf'), '0.1'):
            with self.assertRaises(ValueError):
                validate_config({'good_type_kl_lambda': value, 'good_type_kl_version': VERSION})
        with self.assertRaises(ValueError):
            validate_config({'good_type_kl_lambda': 0.1})
        self.assertEqual(validate_config({}), 0)
        class Tokenizer:
            def encode(self, text, **kwargs):
                return {'<tool_call>': [151657], '<answer': [27, 9217]}[text]
        verify_type_tokenizer(Tokenizer())
        with patch.object(Tokenizer, 'encode', return_value=[27]), self.assertRaises(ValueError):
            verify_type_tokenizer(Tokenizer())
