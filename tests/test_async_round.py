"""Real tiny-Qwen execution with synthetic SPS; no hardware speedup claim."""
import unittest
from unittest.mock import patch

from dspark_qwen.async_capacity import Profile, TwoStepCapacity
from dspark_qwen.async_round import CapacityRoundDriver
from tests import test_packed_sampling as fixtures


class AsyncRoundTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.PackedSamplingTests()
        self.fixture.setUpClass()
        self.fixture.setUp()
        self.fixture.admit(30)
        self.planner = TwoStepCapacity(3)
        self.profile = Profile('synthetic-flat', (1.,) * 12, tuple(range(1, 13)), 0, 128)
        self.driver = CapacityRoundDriver(self.fixture.session, self.planner, self.profile)

    def test_three_real_rounds_order_full_shadow_and_independent_kv(self):
        events = []
        session = self.fixture.session
        original_freeze, original_propose = self.planner.freeze, session.propose
        def freeze(**kwargs):
            events.append('freeze')
            return original_freeze(**kwargs)
        def propose(*args, **kwargs):
            self.assertEqual(events[-1], 'freeze')
            events.append('propose')
            return original_propose(*args, **kwargs)
        with patch.object(self.planner, 'freeze', side_effect=freeze), patch.object(session, 'propose', side_effect=propose):
            for i in range(3):
                result = self.driver.step()
                self.fixture.assert_prefix_content()
                proof, timing = result['policy_validation'], result['capacity_round']
                self.assertEqual(timing['planner_epoch'], i)
                self.assertEqual(proof['history_epoch'], None if i < 2 else 0)
                self.assertEqual(proof['logical_b'], 3 if i < 2 else 12)
                self.assertEqual(sum(x['proposal_positions'] for x in result['work']['proposal_batches']), 9)
                self.assertEqual(result['work']['policy_confidence_host_values'], 9)
                self.assertFalse(proof['hardware_overlap_proven'])
                self.assertGreater(timing['synchronized_wall_seconds'], sum(timing['stage_seconds'].values()))
                self.assertEqual(self.planner.history[i].shadow_positions, 9)
        self.assertEqual(events, ['freeze', 'propose'] * 3)

    def test_existing_draw_and_partial_round_failure_cannot_be_reused(self):
        session = self.fixture.session
        session.propose(list(session.requests))
        with self.assertRaisesRegex(ValueError, 'outstanding'):
            self.driver.step()
        self.assertEqual(self.planner.history, ())
        # Start a fresh independent fixture; do not reuse the pending proposals.
        self.setUp()
        with patch.object(self.fixture.session, 'propose', side_effect=RuntimeError('fixture failure')):
            with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
                self.driver.step()
        with self.assertRaisesRegex(RuntimeError, 'no retry'):
            self.driver.step()


if __name__ == '__main__':
    unittest.main()
