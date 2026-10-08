"""Fraction oracle branch enumeration drives the tensor implementation on CPU."""
from collections import defaultdict
from fractions import Fraction as F
import itertools
import math
import unittest

import torch

from dspark_qwen.sampling import Proposal, verify_proposal as cpu_verify
from dspark_qwen.tensor_sampling import (
    TensorProposal, TensorRandom, check_probabilities, logits_to_probabilities,
    residual_distribution, sample_categorical, verify_proposal,
)
from test_sampling import Tape, proposal_paths, verification_paths


class TensorTape(Tape):
    def uniform(self, reference):
        return torch.tensor(self(), dtype=torch.float64, device=reference.device)


def row(values):
    return torch.tensor([float(v) for v in values], dtype=torch.float64)


def tensor_proposal(tokens, rows, vocab):
    return TensorProposal(torch.tensor(tokens, dtype=torch.long),
        torch.stack([row(r) for r in rows]) if rows else torch.empty((0, vocab), dtype=torch.float64),
        torch.zeros(len(tokens)))


class TensorSamplingTests(unittest.TestCase):
    def test_exact_simplex_law_and_cpu_oracle_every_branch(self):
        simplex = [tuple(F(x, 4) for x in r) for r in
                   itertools.product(range(5), repeat=3) if sum(r) == 4]
        for p, q in itertools.product(simplex, repeat=2):
            law = defaultdict(F)
            for tokens, rows, weight, proposal_tape in proposal_paths(lambda _: q, 1, lambda _: True):
                rng = TensorTape(proposal_tape)
                self.assertEqual(int(sample_categorical(row(q), rng)), tokens[0])
                rng.exhausted()
                for expected, probability, tape in verification_paths(tokens, rows, (p, p), 1, ()):
                    rng = TensorTape(tape)
                    actual = verify_proposal(tensor_proposal(tokens, rows, 3),
                        torch.stack((row(p), row(p))), rng, max_new_tokens=1)
                    rng.exhausted()
                    self.assertEqual(tuple(actual.tokens.tolist()), expected.tokens)
                    self.assertEqual(actual.accepted_draft_tokens, expected.accepted_draft_tokens)
                    self.assertEqual(actual.rejected_index, expected.rejected_index)
                    cpu = cpu_verify(Proposal(tokens, rows), (p, p), Tape(tape), max_new_tokens=1)
                    self.assertEqual(cpu, expected)
                    law[tuple(actual.tokens.tolist())] += weight * probability
            self.assertEqual(dict(law), {(i,): v for i, v in enumerate(p) if v})

    def test_multitoken_eos_budget_and_bonus_match_rational_oracle(self):
        p = lambda prefix: (F(1, 4), F(3, 4)) if not prefix or prefix[-1] else (F(3, 4), F(1, 4))
        q = lambda prefix: (F(1, 2), F(1, 2)) if not prefix else (F(3, 4), F(1, 4))
        for length, budget, stops in itertools.product((0, 1, 2), (0, 1, 2, 3), ((), (1,))):
            for tokens, rows, _, _ in proposal_paths(q, length, lambda _: True):
                target = tuple(p(tokens[:j]) for j in range(len(tokens) + 1))
                for expected, _, tape in verification_paths(tokens, rows, target, budget, stops):
                    rng = TensorTape(tape)
                    actual = verify_proposal(tensor_proposal(tokens, rows, 2),
                        torch.stack([row(r) for r in target]), rng,
                        max_new_tokens=budget, eos_ids=stops)
                    rng.exhausted()
                    self.assertEqual((tuple(actual.tokens.tolist()), actual.accepted_draft_tokens,
                        actual.rejected_index, actual.extra_token_kind, actual.stop_reason),
                        (expected.tokens, expected.accepted_draft_tokens, expected.rejected_index,
                         expected.extra_token_kind, expected.stop_reason))

    def test_probability_adapter_and_tiny_residual(self):
        for dtype in (torch.bfloat16, torch.float32, torch.float64):
            logits = torch.tensor([[0.25, -0.75, 1.5]], dtype=dtype)
            actual = logits_to_probabilities(logits, 0.7)
            expected = (logits.double() / 0.7).softmax(-1)
            expected /= expected.sum(-1, keepdim=True)
            self.assertEqual(actual.dtype, torch.float64)
            self.assertEqual(logits.dtype, dtype)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        delta = 2 ** -40
        torch.testing.assert_close(residual_distribution(row((0.5 + delta, 0.5 - delta)),
                                                         row((0.5, 0.5))), row((1, 0)), atol=0, rtol=0)
        with self.assertRaises(ValueError):
            residual_distribution(row((0.5, 0.5)), row((0.5, 0.5)))
        # Underflowed softmax support is part of this declared numerical law.
        torch.testing.assert_close(logits_to_probabilities(row((1000, -1000)), 1), row((1, 0)))

    def test_strict_boundaries_and_invalid_laws(self):
        self.assertEqual(int(sample_categorical(row((0.5, 0, 0.5)), TensorTape((0.5,)))), 2)
        proposal = tensor_proposal((0,), ((1, 0),), 2)
        actual = verify_proposal(proposal, torch.stack((row((0.5, 0.5)), row((1, 0)))),
                                 TensorTape((0.5, 0)), max_new_tokens=2)
        self.assertEqual(actual.rejected_index, 0)
        self.assertEqual(actual.tokens.tolist(), [1])
        for temperature in (0, -1, math.inf, math.nan, True):
            with self.assertRaises(ValueError):
                logits_to_probabilities(row((0, 1)), temperature)
        for value in (math.nan, math.inf, -math.inf):
            with self.assertRaises(ValueError):
                logits_to_probabilities(row((value, 1)), 1)
        for invalid in (row((0, 0)), row((0.5, 0.6)), row((-0.1, 1.1)), row((math.nan, 1)),
                        torch.tensor([0.5, 0.5], dtype=torch.float32)):
            with self.assertRaises(ValueError):
                check_probabilities(invalid)
        for invalid in (-1, 1, math.inf, math.nan):
            with self.assertRaises(ValueError):
                sample_categorical(row((1, 0)), TensorTape((invalid,)))
        with self.assertRaises(ValueError):
            verify_proposal(tensor_proposal((1,), ((1, 0),), 2),
                            torch.stack((row((0, 1)), row((1, 0)))), TensorTape(()), max_new_tokens=2)

    def test_seeded_generator_is_local_and_repeatable(self):
        torch.manual_seed(19)
        state = torch.random.get_rng_state().clone()
        def samples(seed):
            rng = TensorRandom(torch.Generator().manual_seed(seed))
            return [int(sample_categorical(row((0.25, 0.75)), rng)) for _ in range(20)]
        self.assertEqual(samples(2), samples(2))
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
