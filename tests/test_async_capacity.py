"""Synthetic CPU fixtures, not measured engine profiles or model causality proof."""
from dataclasses import replace
import itertools
import math
import random
import unittest
from types import SimpleNamespace

from dspark_qwen.async_capacity import Profile, Request, TwoStepCapacity, search_capacity


def profile(sps=(1., .5, .45), physical=None):
    return Profile('fixture', tuple(sps), tuple(physical or range(1, len(sps) + 1)), 0, 100)


def freeze(planner, roster, p):
    return planner.freeze(epoch=len(planner.history), session_epoch=10 + len(planner.history), roster=roster, profile=p)


def record(planner, roster, p, rows):
    ticket = freeze(planner, roster, p)
    planner.finish(ticket, rows, shadow_positions=len(roster) * planner.gamma, shadow_seconds=.01)
    return ticket


class AsyncCapacityTests(unittest.TestCase):
    def test_exact_two_steps_and_cliff_full_search(self):
        state = TwoStepCapacity(2); roster = [Request('a', 0, 3, 10)]; p = profile()
        for rows in ([[.8, .9]], [[.8, 0.]]):
            ticket = record(state, roster, p, rows)
            self.assertEqual(ticket.reserved_k, 1)
            self.assertIn('cold_start_target_only_full_shadow', ticket.replanning)
        ticket = freeze(state, roster, p)
        self.assertEqual(ticket.history_epoch, 0)
        self.assertEqual(ticket.historical_k, 3)  # crosses SPS cliff at B=2
        state.finish(ticket, [[0., 0.]], shadow_positions=2, shadow_seconds=.01)
        self.assertEqual(freeze(state, roster, p).historical_k, 1)  # t-2 now frame 1

    def test_historical_search_matches_independent_jagged_exhaustive_oracle(self):
        rng = random.Random(121)
        for r in (1, 2, 3):
            for _ in range(30):
                rows = [[rng.choice([0., .2, .8, 1.]) for _ in range(2)] for _ in range(r)]
                p = profile([rng.uniform(.05, 3) for _ in range(r * 3)])
                roster = [Request(str(i), i, 0, 3) for i in range(r)]
                state = TwoStepCapacity(2); record(state, roster, p, rows)
                candidates = []
                for lengths in itertools.product(range(3), repeat=r):
                    b = r + sum(lengths)
                    expected = r + sum(sum(math.prod(row[:j]) for j in range(1, n + 1)) for row, n in zip(rows, lengths))
                    candidates.append((expected * p.sps[b - 1], -b))
                oracle = max(candidates)
                self.assertEqual(search_capacity(state.history[0]), -oracle[1])

    def test_own_token_later_score_intervention_preserves_own_admission(self):
        # Simulated x_j intervention changes only c_{j+1:}; c_{<=j} fixed.
        # No token values enter the planner. Independent other request fixed.
        state = TwoStepCapacity(3); p = profile([1.] * 5)
        roster = [Request('a', 0, 0, 4), Request('b', 1, 0, 4)]
        for _ in range(2): record(state, roster, p, [[1., 1., 1.], [1., 1., 1.]])
        ticket = freeze(state, roster, p)
        for j in (1, 2, 3):
            for prefix in itertools.product((0., .4, 1.), repeat=j):
                decisions = set()
                for suffix in itertools.product((0., .2, 1.), repeat=3 - j):
                    plan = state.allocate(ticket, [prefix + suffix, [.5, 1., 1.]])
                    decisions.add(plan.lengths[0] >= j)
                self.assertEqual(len(decisions), 1)
        # Changing pre-token c_j itself can legitimately change admission.
        self.assertNotEqual(state.allocate(ticket, [[1., 0., 0.], [.5, 1., 1.]]).lengths[0],
                            state.allocate(ticket, [[0., 0., 0.], [.5, 1., 1.]]).lengths[0])

    def test_ties_canonical_identity_and_zero_underfill(self):
        state = TwoStepCapacity(2); p = profile([1.] * 5, [2, 2, 4, 4, 8])
        roster = [Request('b', 2, 0, 3), Request('a', 1, 0, 3)]
        for _ in range(2): record(state, roster, p, [[1., 1.], [1., 1.]])
        ticket = freeze(state, roster, p)
        plan = state.allocate(ticket, [[1., 1.], [1., 1.]])
        self.assertEqual(plan.lengths, (2, 1))
        self.assertEqual((plan.reserved_k, plan.logical_b, plan.physical_b), (5, 5, 8))
        plan = state.allocate(ticket, [[0., 1.], [0., 1.]])
        self.assertEqual((plan.reserved_k, plan.logical_b, plan.physical_b), (5, 2, 2))
        self.assertEqual(ticket.reservation_physical_b, 8)

    def test_churn_uses_departed_history_and_clamps_known_budget(self):
        state = TwoStepCapacity(2); p = profile([1.] * 6)
        old = [Request('old', 0, 0, 3), Request('same', 1, 0, 3)]
        for _ in range(2): record(state, old, p, [[1., 1.], [1., 1.]])
        ticket = freeze(state, [Request('same', 2, 0, 1)], p)
        self.assertEqual((ticket.historical_k, ticket.reserved_k), (6, 1))
        self.assertEqual(set(ticket.replanning), {'roster_churn', 'exogenous_feasibility_clamp'})
        self.assertEqual(state.allocate(ticket, [[.1, 1.]]).lengths, (0,))
        state.finish(ticket, [[.1, 1.]], shadow_positions=2, shadow_seconds=.01)
        with self.assertRaisesRegex(ValueError, 'Retired'):
            freeze(state, old, p)

    def test_roster_growth_and_empty_rounds(self):
        state = TwoStepCapacity(2); p = profile([1.] * 6)
        for _ in range(2):
            t = freeze(state, [], p)
            self.assertEqual(state.allocate(t, []).logical_b, 0)
            state.finish(t, [], shadow_positions=0, shadow_seconds=0)
        t = freeze(state, [Request('a', 1, 0, 2), Request('b', 2, 0, 2)], p)
        self.assertEqual((t.historical_k, t.reserved_k), (0, 2))
        self.assertIn('exogenous_feasibility_clamp', t.replanning)

    def test_immutability_history_ticket_profile_and_cold_shadow_work(self):
        state = TwoStepCapacity(2); roster = [Request('a', 1, 0, 2)]; p = profile()
        with self.assertRaises(ValueError): state.freeze(epoch=1, session_epoch=1, roster=roster, profile=p)
        t = freeze(state, roster, p)
        with self.assertRaises(ValueError): freeze(state, roster, p)
        with self.assertRaises(ValueError): state.allocate(replace(t), [[1., 1.]])
        with self.assertRaisesRegex(ValueError, 'Full shadow'):
            state.finish(t, [[1., 1.]], shadow_positions=0, shadow_seconds=.01)
        with self.assertRaisesRegex(ValueError, 'charged'):
            state.finish(t, [[1., 1.]], shadow_positions=2, shadow_seconds=0)
        state.finish(t, [[1., 1.]], shadow_positions=2, shadow_seconds=.01)
        with self.assertRaises(ValueError): state.allocate(t, [[1., 1.]])
        record(state, roster, p, [[1., 1.]])
        with self.assertRaisesRegex(ValueError, 'identity'):
            freeze(state, roster, replace(p, name='different'))
        with self.assertRaisesRegex(ValueError, 'Context'):
            freeze(state, [Request('a', 1, 99, 2)], p)

    def test_session_epochs_and_new_incarnations_are_monotonic(self):
        state = TwoStepCapacity(2); p = profile(); roster = [Request('a', 4, 0, 2)]
        record(state, roster, p, [[1., 1.]])
        with self.assertRaisesRegex(ValueError, 'mutation epoch'):
            state.freeze(epoch=1, session_epoch=10, roster=roster, profile=p)
        with self.assertRaisesRegex(ValueError, 'monotonic'):
            freeze(state, [Request('b', 3, 0, 2)], p)
        with self.assertRaisesRegex(ValueError, 'monotonic'):
            freeze(state, [Request('b', 4, 0, 2)], p)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            freeze(state, [Request('a', 4, 0, 2), Request('b', 4, 0, 2)], p)
        record(state, [Request('a', 5, 0, 2)], p, [[1., 1.]])

    def test_bound_capability_rechecks_private_logits_and_pre_token_scope(self):
        state = TwoStepCapacity(2); p = profile(); roster = [Request('a', 1, 0, 3)]
        for _ in range(2): record(state, roster, p, [[.8, .9]])
        ticket = freeze(state, roster, p)
        identity = dict(request='a', incarnation=1, epoch=ticket.session_epoch, cache_length=0,
                        nonce=4, proposal_limit=2, mode='shadow')
        handle = SimpleNamespace(**identity, proposal=SimpleNamespace(confidence_logits=[1., 2.]))
        bound = state.bind(ticket, {'a': handle})
        context = dict(epoch=ticket.session_epoch,
            roster=[dict(request='a', incarnation=1, cache_length=0, remaining_output_budget=3)],
            proposal_identity={'a': dict(identity)}, proposal_confidence_logits={'a': (1., 2.)})
        proof = bound.validate_allocation(context, {'a': handle}, bound.allocations)
        self.assertEqual(proof['history_epoch'], 0)
        self.assertFalse(proof['hardware_overlap_proven'])
        # Changing public observation after binding cannot change the bound plan.
        handle.proposal.confidence_logits[:] = [-10., -10.]
        bound.validate_allocation(context, {'a': handle}, bound.allocations)
        # Binding a tampered observation must fail against the private source.
        tampered = state.bind(ticket, {'a': handle})
        with self.assertRaisesRegex(ValueError, 'confidence source'):
            tampered.validate_allocation(context, {'a': handle}, tampered.allocations)
        with self.assertRaisesRegex(ValueError, 'Foreign'):
            bound.validate_allocation(context, {'a': SimpleNamespace(**vars(handle))}, bound.allocations)
        with self.assertRaisesRegex(ValueError, 'Allocation'):
            bound.validate_allocation(context, {'a': handle}, {'a': 0})
        for key, value in [('epoch', 100), ('roster', []), ('proposal_identity', {}), ('proposal_confidence_logits', {})]:
            with self.assertRaises(ValueError):
                bound.validate_allocation(dict(context, **{key: value}), {'a': handle}, bound.allocations)
        state.finish(ticket, bound.conditional_scores, shadow_positions=2, shadow_seconds=.01)
        with self.assertRaisesRegex(ValueError, 'Stale'):
            bound.validate_allocation(context, {'a': handle}, bound.allocations)

    def test_output_budget_oracle_excludes_no_bonus_last_position(self):
        # Enumerate accept/reject paths, with actual output min(1+A, remaining).
        row = [.8, .9]
        for remaining in (1, 2, 3):
            for length in range(3):
                probability = 1.; expected = 0.
                for rejected in range(length):
                    expected += probability * (1-row[rejected]) * min(rejected+1, remaining)
                    probability *= row[rejected]
                expected += probability * min(length+1, remaining)
                formula = 1 + sum(math.prod(row[:j]) for j in range(1, min(length, remaining-1)+1))
                self.assertAlmostEqual(expected, formula)
            state = TwoStepCapacity(2); p = profile([1., 1., 1.])
            roster = [Request('a', 1, 0, remaining)]
            for _ in range(2): record(state, roster, p, [row])
            t = freeze(state, roster, p)
            self.assertEqual(t.reserved_k, remaining)
            self.assertEqual(state.allocate(t, [row]).lengths, (remaining-1,))
            if remaining == 1:
                self.assertEqual(t.historical_k, 1)  # confidence cannot add progress

    def test_invalid_input_and_insufficient_baseline_capacity(self):
        with self.assertRaises(ValueError): Profile('x', [1], (1,), 0, 4)
        for value in (float('nan'), 0, -1, True):
            with self.assertRaises(ValueError): profile([value])
        state = TwoStepCapacity(2); p = profile([1.])
        with self.assertRaisesRegex(ValueError, 'baseline'):
            freeze(state, [Request('a', 1, 0, 1), Request('b', 2, 0, 1)], p)
        t = freeze(state, [Request('a', 1, 0, 1)], p)
        for rows in ([[1.]], [[1., float('nan')]], [[1., 1.1]], [[True, 1.]]):
            with self.assertRaises(ValueError): state.allocate(t, rows)


if __name__ == '__main__':
    unittest.main()
