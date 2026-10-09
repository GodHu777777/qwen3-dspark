"""Synchronous real-session driver for the CPU two-step capacity reference.

Owns the executable freeze -> propose -> bind -> verify/commit -> finish order.
All host copies, hashing and device synchronizations are inside charged round
wall time. This is not a pipelined or asynchronous execution implementation.
"""
import time

from .async_capacity import Request


class CapacityRoundDriver:
    def __init__(self, session, planner, profile):
        if planner.gamma != session.draft.draft.spec.block_size:
            raise ValueError('Planner gamma must match complete shadow block size')
        self.session, self.planner, self.profile = session, planner, profile
        self.failed = False

    def _synchronize(self):
        if self.session.target.device.type == 'cuda':
            import torch
            torch.cuda.synchronize(self.session.target.device)

    def step(self):
        if self.failed:
            raise RuntimeError('Driver failed; no retry/reuse of partial round')
        if self.session._pending:
            raise ValueError('Driver requires no previously drawn outstanding proposals')
        roster = [Request(name, state['incarnation'], self.session.target.lengths[name],
            state['budget'] - len(state['output']))
            for name, state in self.session.requests.items() if not state['finished']]
        if not roster:
            raise ValueError('No active requests; admit work before starting a round')
        self._synchronize()
        started = time.perf_counter()
        stage_seconds = {}
        try:
            before = time.perf_counter()
            ticket = self.planner.freeze(epoch=len(self.planner.history), session_epoch=self.session.epoch,
                roster=roster, profile=self.profile)
            stage_seconds['freeze'] = time.perf_counter() - before
            before = time.perf_counter()
            batch = self.session.propose([r.request for r in ticket.roster], mode='shadow')
            self._synchronize()
            shadow_seconds = time.perf_counter() - before
            stage_seconds['full_shadow_propose'] = shadow_seconds
            before = time.perf_counter()
            bound = self.planner.bind(ticket, batch.proposals)
            self._synchronize()
            stage_seconds['bind_host_copy_calibration_hash'] = time.perf_counter() - before
            before = time.perf_counter()
            result = self.session.verify_commit(batch.proposals, bound.allocations, allocation_policy=bound)
            self._synchronize()
            stage_seconds['validate_private_source_verify_commit'] = time.perf_counter() - before
            before = time.perf_counter()
            history = self.planner.finish(ticket, bound.conditional_scores,
                shadow_positions=len(roster) * ticket.gamma, shadow_seconds=shadow_seconds)
            stage_seconds['finish_history'] = time.perf_counter() - before
            result['capacity_round'] = dict(planner_epoch=ticket.epoch, session_epoch=ticket.session_epoch,
                history_digest=history.digest, stage_seconds=stage_seconds,
                synchronized_wall_seconds=time.perf_counter() - started,
                timing_scope='Synchronous full round, including shadow work, host copies, calibration, hashing and verification',
                ordering='freeze -> full shadow propose -> bind -> validate/verify/commit -> finish',
                hardware_overlap_proven=False)
            return result
        except Exception:
            self.failed = True
            raise
