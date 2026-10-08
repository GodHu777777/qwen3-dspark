"""Independent exhaustive allocation oracles; fixtures are NOT hardware profiles."""
import itertools
import math
import random
import unittest

from dspark_qwen.scheduler import plan_prefixes


def exhaustive(confidences, sps, capacity, exact_batch=None):
    """Enumerate every feasible ell vector; deliberately no sorted greedy path."""
    candidates = []
    for lengths in itertools.product(*(range(len(row) + 1) for row in confidences)):
        batch = len(confidences) + sum(lengths)
        if batch > capacity:
            continue
        if exact_batch is not None and batch != exact_batch:
            continue
        expected = sum(1 + sum(math.prod(row[:j]) for j in range(1, length + 1))
            for row, length in zip(confidences, lengths))
        candidates.append((expected * sps[batch], expected, batch, lengths))
    return max(candidates, key=lambda candidate: candidate[0])


class SchedulerTests(unittest.TestCase):
    def test_global_shared_budget_matches_exhaustive_allocation_at_every_capacity(self):
        rng = random.Random(20261008)
        cases = [[[0.9, 0.9], [0.8, 0.99]], [[1, 1, 1], [1, 1, 1]], [[0, 1], [0.7, 0]]]
        for requests in (1, 2, 3):
            for length in (1, 2, 3, 4):
                cases.append([[rng.random() for _ in range(length)] for _ in range(requests)])
        for confidence in cases:
            r, length = len(confidence), len(confidence[0])
            curve = {batch: 1.0 for batch in range(r, r * (length + 1) + 1)}
            for capacity in curve:
                with self.subTest(confidence=confidence, capacity=capacity):
                    plan = plan_prefixes(confidence, curve, max_batch_tokens=capacity)
                    oracle = exhaustive(confidence, curve, capacity)
                    self.assertAlmostEqual(plan.estimated_tokens_per_second, oracle[0], places=12)
                    self.assertEqual(plan.target_batch_tokens, r + sum(plan.prefix_lengths))
                    self.assertLessEqual(plan.target_batch_tokens, capacity)
        # A uniform per-request k=1 is suboptimal: the first request's second
        # prefix has more survival mass than the second request's first token.
        plan = plan_prefixes(cases[0], {2: 1, 3: 1, 4: 1}, max_batch_tokens=4)
        self.assertEqual(plan.prefix_lengths, (2, 0))

    def test_smooth_convex_time_curves_match_true_global_maximum(self):
        # Concave expected progress along admissions / positive convex step
        # time gives a unimodal ratio. Enumerate ell vectors independently.
        rng = random.Random(49)
        for requests in (1, 2, 3):
            for length in (1, 2, 3, 4):
                for _ in range(8):
                    confidence = [[rng.uniform(0.05, 1) for _ in range(length)] for _ in range(requests)]
                    fixed, linear, quadratic = rng.uniform(0.1, 8), rng.uniform(0, 1), rng.uniform(0, 0.4)
                    maximum = requests * (length + 1)
                    curve = {batch: 1 / (fixed + linear * batch + quadratic * batch ** 2)
                        for batch in range(requests, maximum + 1)}
                    plan = plan_prefixes(confidence, curve)
                    oracle = exhaustive(confidence, curve, maximum)
                    self.assertAlmostEqual(plan.estimated_tokens_per_second, oracle[0], places=12)

    def test_each_admission_maximizes_expected_progress_for_its_exact_batch(self):
        confidence = [[0.91, 0.3, 0.95], [0.85, 0.99, 0.1], [1.0, 0.45, 0.9]]
        curve = {batch: 1.0 for batch in range(3, 13)}
        plan = plan_prefixes(confidence, curve)
        for admission in plan.admissions:
            batch = admission.trial_batch_tokens
            oracle = exhaustive(confidence, curve, batch, exact_batch=batch)
            self.assertEqual(oracle[2], batch)  # Exact B, not merely <= capacity.
            self.assertAlmostEqual(admission.trial_expected_tokens, oracle[1], places=12)

    def test_literal_positive_candidate_filter_on_rising_sps_curve(self):
        # Algorithm 1 line 4 excludes zero survival. A rising curve could make
        # extra zero-gain tokens beneficial to score; we report literal behavior
        # rather than claiming a global optimum on arbitrary hardware curves.
        curve = {1: 1.0, 2: 100.0, 3: 100.0}
        plan = plan_prefixes([[0.0, 1.0]], curve)
        self.assertEqual(plan.prefix_lengths, (0,))
        self.assertEqual(plan.estimated_tokens_per_second, 1.0)
        self.assertEqual(plan.admissions, ())
        self.assertEqual(exhaustive([[0.0, 1.0]], curve, 3)[0], 100.0)

    def test_paper_cliff_counterexample_preserves_early_stop_and_reports_nonoptimality(self):
        curve = {1: 1.0, 2: 0.5, 3: 0.45}
        plans = [plan_prefixes([[0.8, future]], curve) for future in (0.9, 0.0)]
        for plan in plans:
            self.assertEqual(plan.prefix_lengths, (0,))
            self.assertEqual(plan.stopping_reason, "first_non_improvement")
            self.assertEqual(len(plan.admissions), 1)
            self.assertFalse(plan.admissions[0].admitted)
            self.assertAlmostEqual(plan.admissions[0].trial_tokens_per_second, 0.9)
        # Hindsight changes first-token admission depending on c2 (which may
        # depend on sampled x1): exactly the Appendix A selection-bias trap.
        high = exhaustive([[0.8, 0.9]], curve, 3)
        low = exhaustive([[0.8, 0.0]], curve, 3)
        self.assertEqual(high[3], (2,))
        self.assertEqual(low[3], (0,))
        self.assertAlmostEqual(high[0], 1.134)
        self.assertGreater(high[0], plans[0].estimated_tokens_per_second)

    def test_probability_ties_zeroes_and_baseline_accounting(self):
        plan = plan_prefixes([[1, 1, 1], [1, 1, 1]], {2: 4, 3: 4, 4: 4, 5: 4}, max_batch_tokens=5)
        self.assertEqual(plan.prefix_lengths, (3, 0))
        self.assertEqual([(x.request, x.position) for x in plan.admissions], [(0, 1), (0, 2), (0, 3)])
        self.assertEqual(plan.expected_tokens, 5)
        self.assertEqual(plan.estimated_tokens_per_second, 20)  # SPS is steps/s.
        empty = plan_prefixes([], {})
        self.assertEqual(empty.stopping_reason, "no_active_requests")
        self.assertEqual(empty.target_batch_tokens, 0)
        target_only = plan_prefixes([[], []], {2: 10})
        self.assertEqual(target_only.expected_tokens, 2)
        self.assertEqual(target_only.estimated_tokens_per_second, 20)
        zeros = plan_prefixes([[0, 1]], {1: 1, 2: 1, 3: 1})
        self.assertEqual(zeros.prefix_lengths, (0,))
        self.assertEqual(zeros.admissions, ())

    def test_invalid_or_incomplete_capacity_profiles_are_not_interpolated(self):
        with self.assertRaisesRegex(ValueError, "Missing discrete SPS"):
            plan_prefixes([[0.8, 0.8]], {1: 1, 3: 0.9})
        with self.assertRaisesRegex(ValueError, "baseline token"):
            plan_prefixes([[0.9], [0.8]], {}, max_batch_tokens=1)
        for value in (-0.1, 1.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError): plan_prefixes([[value]], {1: 1, 2: 1})
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError): plan_prefixes([[0.8]], {1: 1, 2: value})
        with self.assertRaises(ValueError): plan_prefixes([[0.8], [0.9, 0.8]], {2: 1, 3: 1, 4: 1, 5: 1})
        with self.assertRaises(ValueError): plan_prefixes([[0.8]], {1: 1, 2: 1}, max_batch_tokens=True)


if __name__ == "__main__":
    unittest.main()
