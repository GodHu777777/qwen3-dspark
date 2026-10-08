import math
import unittest
from dataclasses import replace

from dspark_qwen.calibration import (
    RolloutBlock, calibrated_probabilities, fit_sts as _fit_sts, reliability_metrics,
)


IDENTITY = dict(checkpoint_sha256='a'*64, development_records_sha256='b'*64,
    rollout_protocol_sha256='c'*64, probability_policy='synthetic float64 stochastic fixture')


def fit_sts(*args, **kwargs):
    return _fit_sts(*args, identity=IDENTITY, **kwargs)


def block(name, accepted, *, length=2, logits=None, eos=None):
    return RolloutBlock(name, 'dev-'+name, 'validation',
        tuple(logits if logits is not None else [math.log(4)]*length),
        length, accepted, length, eos)


class CalibrationTests(unittest.TestCase):
    def test_sequential_joint_objective_exact_fixture(self):
        # Five blocks: P(prefix 1)=4/5, P(prefix 2)=3/5.
        # c1=.8 and c2=.75 is exact; independently fitting c2=.6 is wrong.
        rows = [block(str(i), accepted) for i, accepted in enumerate([2, 2, 2, 1, 0])]
        t2 = math.log(4)/math.log(3)
        fit = fit_sts(rows, block_size=2, temperature_grid=[1, t2, 2])
        self.assertEqual(fit['temperatures'], [1, t2])
        for entry in fit['per_position']:
            self.assertEqual(entry['count'], 5)
            self.assertAlmostEqual(entry['ece'], 0)
        self.assertAlmostEqual(fit['per_position'][1]['brier'], .24)
        self.assertAlmostEqual(fit['uncalibrated_per_position'][1]['ece'], .04)

    def test_scale_logits_before_cumulative_product(self):
        conditional, prefix = calibrated_probabilities([math.log(9), math.log(9)], [2, 1])
        self.assertAlmostEqual(conditional[0], .75)
        self.assertAlmostEqual(prefix[1], .675)
        self.assertNotAlmostEqual(prefix[1], math.sqrt(.9*.9))

    def test_eos_tail_and_rejection_denominators(self):
        rows = [block('eos', 3, length=3, eos=0),
                block('rejected', 0, length=3), block('budget', 1, length=1),
                block('none', 0, length=0),
                replace(block('unverified', 1, length=3), verified_length=1)]
        fit = fit_sts(rows, block_size=4, temperature_grid=[1])
        self.assertEqual([x['count'] for x in fit['per_position']], [4, 1, 1, 0])
        self.assertEqual([x['target_mean'] for x in fit['per_position']], [3/4, 0, 0, None])
        self.assertFalse(fit['per_position'][3]['fitted'])
        self.assertEqual(fit['temperatures'][3], 1)

    def test_bin_edges_clamp_and_weighted_ece(self):
        metrics = reliability_metrics([0, .5, 1], [0, 1, 1], num_bins=2)
        self.assertEqual([b['count'] for b in metrics['bins']], [1, 2])
        self.assertAlmostEqual(metrics['ece'], (.5+2e-8)/3)
        self.assertAlmostEqual(metrics['brier'], (.25+2e-16)/3)
        self.assertIsNone(reliability_metrics([], [])['ece'])

    def test_ties_grid_order_and_extreme_logits(self):
        rows = [block('a', 0, length=1, logits=[0]), block('b', 1, length=1, logits=[0])]
        self.assertEqual(fit_sts(rows, block_size=1, temperature_grid=[2, .5, 1])['temperatures'], [1])
        self.assertEqual(calibrated_probabilities([-1e300, 1e300], [1e-300, 1e-300]), ([0, 1], [0, 0]))

    def test_validation_and_uncensored_sampling_boundary(self):
        good = block('a', 1)
        for bad in [replace(good, split='train'), replace(good, split='test'),
                    replace(good, split='final-test'), replace(good, sampling_mode='greedy'),
                    replace(good, collection_policy='confidence_threshold'),
                    replace(good, proposal_length=3), replace(good, verified_length=3), replace(good, accepted_prefix_length=3),
                    replace(good, accepted_eos_position=1), replace(good, confidence_logits=(float('nan'),0))]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                fit_sts([bad], block_size=2)
        with self.assertRaises(ValueError):
            fit_sts([good, good], block_size=2)
        with self.assertRaises(ValueError):
            fit_sts([], block_size=2)
        greedy = replace(good, sampling_mode='greedy')
        self.assertEqual(fit_sts([greedy], block_size=2, sampling_mode='greedy')['sampling_mode'], 'greedy')

    def test_required_frozen_identity(self):
        for identity in ({}, {**IDENTITY, 'probability_policy': ''},
                         {**IDENTITY, 'checkpoint_sha256': 'wrong'}):
            with self.subTest(identity=identity), self.assertRaises(ValueError):
                _fit_sts([block('a',1)], block_size=2, identity=identity)
        fit = fit_sts([block('a',1)], block_size=2)
        self.assertEqual(fit['identity'], IDENTITY)

    def test_invalid_grid_bins_and_soft_labels(self):
        for grid in ([], [0], [-1], [float('inf')], [True], [.5, 2]):
            with self.subTest(grid=grid), self.assertRaises(ValueError):
                fit_sts([block('a',1)], block_size=2, temperature_grid=grid)
        for bins in (0, -1, 1.5, True):
            with self.subTest(bins=bins), self.assertRaises(ValueError):
                reliability_metrics([], [], num_bins=bins)
        with self.assertRaises(ValueError):
            reliability_metrics([.5], [.5])


if __name__ == '__main__':
    unittest.main()
