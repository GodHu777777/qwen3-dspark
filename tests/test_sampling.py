"""Exact rational path enumeration, exercised against the float CPU sampler.

No Monte Carlo tolerance is used for the probability-law assertions. Fractions
enumerate every proposal/accept/reject/correction/bonus branch. Each branch also
drives the implementation with interior points of its random-number intervals.
"""
from collections import defaultdict
from fractions import Fraction as F
import itertools
import math
import random
import unittest

from dspark_qwen.sampling import (
    Proposal, SampledRound, acceptance_probability, probabilities,
    residual_distribution, sample_categorical, sample_proposal, verify_proposal,
)


def categorical_branches(row):
    start = F(0)
    for token, mass in enumerate(row):
        if mass:
            yield token, mass, start + mass / 2
        start += mass
    assert start == 1


class Tape:
    def __init__(self, values):
        self.values = list(values)
        self.used = 0

    def __call__(self):
        if self.used == len(self.values):
            raise AssertionError("Unexpected random draw")
        result = float(self.values[self.used])
        self.used += 1
        return result

    def exhausted(self):
        assert self.used == len(self.values), (self.used, len(self.values))


def proposal_paths(q, gamma, admit, prefix=(), rows=(), mass=F(1), tape=()):
    if len(prefix) == gamma or not admit(prefix):
        yield prefix, rows, mass, tape
        return
    row = q(prefix)
    for token, probability, midpoint in categorical_branches(row):
        yield from proposal_paths(q, gamma, admit, prefix + (token,),
                                  rows + (row,), mass * probability, tape + (midpoint,))


def verification_paths(tokens, q, p, budget, stops, j=0, tape=(), mass=F(1)):
    """Independent rational oracle, stopping before unused random draws."""
    if j == budget or (j and tokens[j - 1] in stops):
        reason = "eos" if j and tokens[j - 1] in stops else "budget"
        yield SampledRound(tokens[:j], j, None, None, reason), mass, tape
        return
    if j == len(tokens):
        for token, probability, midpoint in categorical_branches(p[j]):
            output = tokens + (token,)
            reason = "eos" if token in stops else (
                "budget" if len(output) == budget else "round_complete")
            yield SampledRound(output, j, None, "bonus", reason), mass * probability, tape + (midpoint,)
        return
    alpha = min(F(1), p[j][tokens[j]] / q[j][tokens[j]])
    if alpha:
        yield from verification_paths(tokens, q, p, budget, stops, j + 1,
                                      tape + (alpha / 2,), mass * alpha)
    if alpha < 1:
        positive = tuple(max(pi - qi, F(0)) for pi, qi in zip(p[j], q[j]))
        rejection_mass = sum(positive)
        assert rejection_mass > 0
        residual = tuple(x / rejection_mass for x in positive)
        for token, probability, midpoint in categorical_branches(residual):
            output = tokens[:j] + (token,)
            reason = "eos" if token in stops else (
                "budget" if len(output) == budget else "round_complete")
            yield SampledRound(output, j, j, "residual", reason), (
                mass * (1 - alpha) * probability), tape + ((1 + alpha) / 2, midpoint)


def exact_round(p, q, gamma, budget, stops=(), admit=lambda prefix: True):
    """Return exact emitted-block law; test implementation on every branch."""
    result = defaultdict(F)
    for tokens, rows, proposal_mass, proposal_tape in proposal_paths(q, gamma, admit):
        rng = Tape(proposal_tape)
        proposal = sample_proposal(q, gamma, rng, admit=admit)
        rng.exhausted()
        assert proposal.tokens == tokens
        target_rows = tuple(p(tokens[:j]) for j in range(len(tokens) + 1))
        conditional_mass = F(0)
        for expected, verification_mass, tape in verification_paths(
                tokens, rows, target_rows, budget, stops):
            rng = Tape(tape)
            actual = verify_proposal(proposal, target_rows, rng,
                                     max_new_tokens=budget, stop_token_ids=stops)
            rng.exhausted()
            assert actual == expected, (actual, expected)
            result[actual.tokens] += proposal_mass * verification_mass
            conditional_mass += verification_mass
        assert conditional_mass == 1
    assert sum(result.values()) == 1
    return dict(result)


def exact_speculative_sequence(p, q, gamma, budget, stops=(), admit=lambda prefix: True):
    result = defaultdict(F)

    def extend(prefix, mass):
        if len(prefix) == budget or (prefix and prefix[-1] in stops):
            result[prefix] += mass
            return
        law = exact_round(lambda draft: p(prefix + draft), lambda draft: q(prefix + draft),
                          gamma, budget - len(prefix), stops, admit)
        for block, probability in law.items():
            extend(prefix + block, mass * probability)

    extend((), F(1))
    return dict(result)


def exact_target_sequence(p, budget, stops=(), prefix=()):
    if len(prefix) == budget or (prefix and prefix[-1] in stops):
        return {prefix: F(1)}
    result = {}
    for token, probability in enumerate(p(prefix)):
        if probability:
            for continuation, conditional in exact_target_sequence(p, budget, stops, prefix + (token,)).items():
                result[continuation] = probability * conditional
    return result


class SamplingTests(unittest.TestCase):
    def test_exact_one_token_law_on_rational_simplex_including_zeros(self):
        simplex = [tuple(F(x, 4) for x in row) for row in
                   itertools.product(range(5), repeat=3) if sum(row) == 4]
        for p, q in itertools.product(simplex, repeat=2):
            law = exact_round(lambda _: p, lambda _: q, 1, 1)
            self.assertEqual(law, {(token,): mass for token, mass in enumerate(p) if mass})

    def test_exact_autoregressive_sequence_eos_budget_and_adaptive_admission(self):
        # True prefix dependence; q is generally different from p, with zeros.
        p_rows = ((F(1, 2), F(1, 3), F(1, 6)),
                  (F(0), F(1, 4), F(3, 4)), (F(2, 3), F(1, 3), F(0)))
        q_rows = ((F(1, 4), F(3, 4), F(0)),
                  (F(1, 2), F(0), F(1, 2)), (F(1, 6), F(1, 3), F(1, 2)))
        p = lambda prefix: p_rows[(sum(prefix) + len(prefix)) % 3]
        q = lambda prefix: q_rows[(sum(prefix) + 2 * len(prefix)) % 3]
        for gamma, budget, stops in itertools.product((0, 1, 2), (0, 1, 3), ((), (2,))):
            for admit in (lambda prefix: True, lambda prefix: not prefix or prefix[-1] == 0):
                with self.subTest(gamma=gamma, budget=budget, stops=stops):
                    actual = exact_speculative_sequence(p, q, gamma, budget, stops, admit)
                    self.assertEqual(actual, exact_target_sequence(p, budget, stops))
                    self.assertEqual(sum(actual.values()), 1)

    def test_equal_distributions_all_accept_and_bonus(self):
        row = (F(1, 3), F(2, 3))
        self.assertEqual(residual_distribution((1, 0), (0, 1)), (1, 0))
        with self.assertRaises(ValueError):
            residual_distribution(row, row)
        for token in range(2):
            self.assertEqual(acceptance_probability(row, row, token), 1)
        self.assertEqual(exact_round(lambda _: row, lambda _: row, 2, 3),
                         exact_target_sequence(lambda _: row, 3))

    def test_disjoint_support_first_rejection_and_zero_q(self):
        proposal = Proposal((1, 1), ((0, 1), (0, 1)))
        rng = Tape((0, F(1, 2)))  # alpha=0 must reject even u=0.
        actual = verify_proposal(proposal, ((1, 0),) * 3, rng, max_new_tokens=3)
        rng.exhausted()
        self.assertEqual(actual, SampledRound((0,), 0, 0, "residual", "round_complete"))
        with self.assertRaises(ValueError):
            acceptance_probability((1, 0), (0, 1), 0)
        with self.assertRaises(ValueError):
            verify_proposal(Proposal((0,), ((0, 1),)), ((1, 0),) * 2, Tape(()), max_new_tokens=2)

    def test_eos_in_accepted_residual_bonus_and_rejected_eos(self):
        cases = [
            (Proposal((1, 0), ((0, 1), (1, 0))), ((0, 1), (1, 0), (1, 0)),
             (F(1, 2),), SampledRound((1,), 1, None, None, "eos")),
            (Proposal((0,), ((1, 0),)), ((0, 1), (1, 0)),
             (F(1, 2), F(1, 2)), SampledRound((1,), 0, 0, "residual", "eos")),
            (Proposal((), ()), ((0, 1),),
             (F(1, 2),), SampledRound((1,), 0, None, "bonus", "eos")),
            (Proposal((1,), ((0, 1),)), ((1, 0), (0, 1)),
             (F(1, 2), F(1, 2)), SampledRound((0,), 0, 0, "residual", "round_complete")),
        ]
        for proposal, p, values, expected in cases:
            rng = Tape(values)
            self.assertEqual(verify_proposal(proposal, p, rng, max_new_tokens=3, stop_token_ids=(1,)), expected)
            rng.exhausted()

    def test_budget_truncation_consumes_no_unused_randomness(self):
        proposal = Proposal((0, 0), ((1, 0),) * 2)
        for budget in (0, 1, 2):
            rng = Tape([0.5] * budget)
            self.assertEqual(verify_proposal(proposal, ((1, 0),) * 3, rng, max_new_tokens=budget),
                             SampledRound((0,) * budget, budget, None, None, "budget"))
            rng.exhausted()

    def test_tiny_residual_has_no_epsilon_fallback(self):
        delta = 2 ** -40
        p, q = (0.5 + delta, 0.5 - delta), (0.5, 0.5)
        self.assertEqual(residual_distribution(p, q), (1, 0))
        alpha = acceptance_probability(p, q, 1)
        rng = Tape(((1 + alpha) / 2, 0.7))
        self.assertEqual(verify_proposal(Proposal((1,), (q,)), (p, p), rng,
                                        max_new_tokens=2).tokens, (0,))
        rng.exhausted()

    def test_strict_acceptance_and_cdf_boundaries(self):
        # Equality u=alpha rejects, including alpha=0; alpha=1 always accepts.
        proposal = Proposal((0,), ((1, 0),))
        rng = Tape((F(1, 2), 0))
        actual = verify_proposal(proposal, ((0.5, 0.5), (1, 0)), rng, max_new_tokens=2)
        self.assertEqual(actual, SampledRound((1,), 0, 0, "residual", "round_complete"))
        rng.exhausted()
        self.assertEqual(sample_categorical((0.5, 0, 0.5), lambda: 0.5), 2)
        rng = Tape((math.nextafter(1, 0),))
        self.assertEqual(verify_proposal(proposal, ((1, 0), (1, 0)), rng,
                                        max_new_tokens=1).accepted_draft_tokens, 1)
        rng.exhausted()

    def test_normalization_nonfinite_and_invalid_inputs(self):
        for row in ((), (0, 0), (2, 1), (-0.1, 1.1), (0.2, 0.2),
                    (math.nan, 1), (math.inf, 0), (-math.inf, 1), (True, 0), ("1", 0), (10**1000, 0)):
            with self.subTest(row=row), self.assertRaises(ValueError):
                probabilities(row)
        self.assertAlmostEqual(sum(probabilities((0.5, 0.5 + 1e-13))), 1)
        for row in ((0.5, 0.5 + 1e-9),):
            with self.assertRaises(ValueError):
                probabilities(row)
        for invalid in (-1, 1, math.nan, math.inf, True, 10**1000):
            with self.assertRaises(ValueError):
                sample_categorical((1, 0), lambda: invalid)
        self.assertEqual(sample_categorical((0, 1, 0), lambda: 0), 1)
        self.assertEqual(sample_categorical((0, 1, 0), lambda: math.nextafter(1, 0)), 1)
        for invalid in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                sample_proposal(lambda _: (1,), invalid, Tape(()))
            with self.assertRaises(ValueError):
                verify_proposal(Proposal((), ()), ((1,),), Tape(()), max_new_tokens=invalid)
        bad_cases = [
            (Proposal((0,), ()), ((1,), (1,))),
            (Proposal((), ()), ()),
            (Proposal((0,), ((1,),)), ((1,),)),
            (Proposal((0,), ((1,),)), ((1,), (1, 0))),
            (Proposal((True,), ((1,),)), ((1,), (1,))),
            (Proposal((1,), ((1,),)), ((1,), (1,))),
            (Proposal((0,), ((1,),)), ((1,), (math.nan,))),
        ]
        for proposal, p in bad_cases:
            with self.assertRaises(ValueError):
                verify_proposal(proposal, p, Tape(()), max_new_tokens=0)
        with self.assertRaises(ValueError):
            verify_proposal(Proposal((), ()), ((1,),), Tape(()), max_new_tokens=0, stop_token_ids=(-1,))
        with self.assertRaises(ValueError):
            verify_proposal(Proposal((), ()), ((1,),), Tape(()), max_new_tokens=0, stop_token_ids=(0, False))
        with self.assertRaises(ValueError):
            residual_distribution((1,), (1, 0))
        with self.assertRaises(ValueError):
            sample_proposal(lambda prefix: (1,) if not prefix else (1, 0), 2, Tape((0.5,)))
        with self.assertRaises(ValueError):
            sample_proposal(lambda _: (1,), 1, Tape(()), admit=lambda _: 1)

    def test_admission_before_draw_and_anticipating_counterexample(self):
        seen = []
        def admit(prefix):
            seen.append(prefix)
            return len(prefix) < 1
        rng = Tape((F(1, 2),))
        proposal = sample_proposal(lambda _: (1, 0), 3, rng, admit=admit)
        rng.exhausted()
        self.assertEqual(proposal.tokens, (0,))
        self.assertEqual(seen, [(), (0,)])
        # Forbidden hindsight admission: admit X only if X=0; otherwise draw p.
        # Even p=q cannot repair the selection bias with ordinary verification.
        biased = defaultdict(F)
        for candidate in (0, 1):
            for fallback in (0, 1):
                output = candidate if candidate == 0 else fallback
                biased[output] += F(1, 4)
        self.assertEqual(dict(biased), {0: F(3, 4), 1: F(1, 4)})
        self.assertNotEqual(dict(biased), {0: F(1, 2), 1: F(1, 2)})

    def test_seeded_random_api_smoke(self):
        def run(seed):
            rng = random.Random(seed).random
            proposal = sample_proposal(lambda _: (0.25, 0.75), 3, rng)
            return verify_proposal(proposal, ((0.75, 0.25),) * 4, rng, max_new_tokens=4)
        self.assertEqual(run(123), run(123))


if __name__ == "__main__":
    unittest.main()
