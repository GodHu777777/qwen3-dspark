"""CPU two-step capacity reference, separate from synchronous Algorithm 1.

No streams, graphs, physical execution or measured speedup are implemented.
Call freeze BEFORE any current proposal draw; allocate only after full shadow
proposals. Record every epoch, including cold-start/zero-admission epochs.
Numeric score inputs cannot prove model-level pre-token provenance.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Real

from .calibration import calibrated_probabilities


def _integer(x, name, minimum=0):
    if type(x) is not int or x < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def _digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True, order=True)
class Request:
    request: str
    incarnation: int
    cache_length: int
    remaining_output_budget: int

    def __post_init__(self):
        if not isinstance(self.request, str) or not self.request:
            raise ValueError('Nonempty request identity required')
        _integer(self.incarnation, 'incarnation')
        _integer(self.cache_length, 'cache_length')
        _integer(self.remaining_output_budget, 'remaining_output_budget', 1)


@dataclass(frozen=True)
class Profile:
    """Exact discrete SPS and physical bucket for logical B=1..len(sps).

    Context interval conservatively covers cache_length + anchor + full shadow
    gamma for every current request. Changing this interval or any curve value
    changes the profile identity; an incompatible t-2 profile requires a new
    planner (explicit cold start), not silently mixing measurements.
    """
    name: str
    sps: tuple[float, ...]
    physical: tuple[int, ...]
    context_min: int
    context_max: int

    def __post_init__(self):
        if not self.name or type(self.sps) is not tuple or type(self.physical) is not tuple:
            raise ValueError('Named immutable tuple profile required')
        if not self.sps or len(self.sps) != len(self.physical):
            raise ValueError('Complete discrete SPS/physical shape domain required')
        _integer(self.context_min, 'context_min')
        _integer(self.context_max, 'context_max', self.context_min)
        for b, (sps, physical) in enumerate(zip(self.sps, self.physical), 1):
            if isinstance(sps, bool) or not isinstance(sps, Real) or not math.isfinite(sps) or sps <= 0:
                raise ValueError('SPS must be finite and positive')
            _integer(physical, 'physical bucket', b)

    @property
    def identity(self):
        return _digest(asdict(self))


@dataclass(frozen=True)
class Calibration:
    temperatures: tuple[float, ...]
    source_digest: str

    def __post_init__(self):
        if type(self.temperatures) is not tuple or not self.temperatures:
            raise ValueError('Immutable nonempty temperature tuple required')
        if not isinstance(self.source_digest, str) or len(self.source_digest) != 64 or any(c not in '0123456789abcdef' for c in self.source_digest):
            raise ValueError('Calibration source SHA256 required')
        calibrated_probabilities([], self.temperatures)

    @property
    def identity(self):
        return _digest(asdict(self))


@dataclass(frozen=True)
class Capacity:
    epoch: int  # consecutive planner round, NOT model cache mutation epoch
    session_epoch: int
    roster: tuple[Request, ...]
    profile: Profile
    history_epoch: int | None
    history_digest: str | None
    historical_k: int
    reserved_k: int
    reservation_physical_b: int
    replanning: tuple[str, ...]
    gamma: int
    calibration: Calibration


@dataclass(frozen=True)
class Allocation:
    lengths: tuple[int, ...]  # follows canonical ticket roster
    logical_b: int
    physical_b: int
    reserved_k: int
    score_digest: str


@dataclass(frozen=True)
class History:
    epoch: int
    roster: tuple[Request, ...]
    profile: Profile
    scores: tuple[tuple[float, ...], ...]  # complete shadow cumulative scores
    shadow_positions: int
    shadow_seconds: float
    digest: str


def _scores(rows, count, gamma):
    rows = tuple(tuple(row) for row in rows)
    if len(rows) != count or any(len(row) != gamma for row in rows):
        raise ValueError('Complete full-shadow conditional score rows required')
    result = []
    for row in rows:
        survival, cumulative = 1., []
        for c in row:
            if isinstance(c, bool) or not isinstance(c, Real) or not math.isfinite(c) or not 0 <= c <= 1:
                raise ValueError('Conditional score must be finite in [0, 1]')
            survival *= float(c)
            cumulative.append(survival)
        result.append(tuple(cumulative))
    return tuple(result)


def _rank(roster, scores, *, budgeted):
    # Tie ordering is independent of roster insertion order and sampled tokens;
    # earlier own positions always precede later own positions, including ties.
    candidates = []
    for i, (request, row) in enumerate(zip(roster, scores)):
        limit = min(len(row), max(0, request.remaining_output_budget - 1)) if budgeted else len(row)
        for j, score in enumerate(row[:limit], 1):
            candidates.append((score, request.request, request.incarnation, j, i))
    return sorted(candidates, key=lambda x: (-x[0], x[1], x[2], x[3]))


def search_capacity(history):
    """Full historical search across every feasible integer B, no early stop.

    Historical roster includes departed requests. Zero-score extensions remain
    in this search: a rising SPS cliff can favor their reserved shape. Current
    admission excludes zero scores, so reservation and execution can differ.
    """
    r = len(history.roster)
    if not r:
        return 0
    profile = history.profile
    if r > len(profile.sps):
        raise ValueError('Historical roster exceeds profile capacity')
    expected = float(r)
    best_k, best = r, expected * profile.sps[r - 1]
    if not math.isfinite(best):
        raise ValueError('Capacity objective overflow')
    for b, candidate in enumerate(_rank(history.roster, history.scores, budgeted=True), r + 1):
        if b > len(profile.sps):
            break
        expected += candidate[0]
        score = expected * profile.sps[b - 1]
        if not math.isfinite(score):
            raise ValueError('Capacity objective overflow')
        if score > best:  # exact ties preserve smaller K
            best_k, best = b, score
    return best_k


class TwoStepCapacity:
    """Single-owner sequential reference; frozen issued tickets are capabilities.

    No skipped/overwritten/future history frames. Session incarnations may never
    be reused after retirement. Work counters are caller observations, not
    independently measured hardware accounting. Full shadow work is mandatory.
    """
    def __init__(self, gamma, *, calibration=None):
        _integer(gamma, 'gamma', 1)
        self.gamma = gamma
        self._calibration = calibration or Calibration((1.,) * gamma, _digest({'transform': 'unscaled-sigmoid-v1'}))
        if not isinstance(self._calibration, Calibration) or len(self._calibration.temperatures) != gamma:
            raise ValueError('Exactly gamma frozen temperatures required')
        self._frames = []
        self._pending = None
        self._seen = set()
        self._previous = set()
        self._last_session_epoch = -1

    @property
    def calibration(self):
        return self._calibration

    @property
    def history(self):
        return tuple(self._frames)

    def freeze(self, *, epoch, session_epoch, roster, profile):
        _integer(epoch, 'epoch')
        _integer(session_epoch, 'session_epoch')
        if self._pending is not None or epoch != len(self._frames):
            raise ValueError('Complete previous epoch; no skipped, future or overwritten history')
        if not isinstance(profile, Profile):
            raise ValueError('Frozen Profile required')
        roster = tuple(roster)
        if any(not isinstance(r, Request) for r in roster):
            raise ValueError('Request records required')
        roster = tuple(sorted(roster))
        if session_epoch <= self._last_session_epoch:
            raise ValueError('Session mutation epoch must advance between planner rounds')
        if len({r.request for r in roster}) != len(roster) or len({r.incarnation for r in roster}) != len(roster):
            raise ValueError('Duplicate request or incarnation')
        keys = {(r.request, r.incarnation) for r in roster}
        if any(key in self._seen and key not in self._previous for key in keys):
            raise ValueError('Retired request incarnation cannot be reused')
        previous_max = max((inc for _, inc in self._seen), default=-1)
        if any(inc <= previous_max for name, inc in keys - self._previous):
            raise ValueError('New incarnation must be globally monotonic and never reused')
        if len(roster) > len(profile.sps):
            raise ValueError('Capacity cannot fit all baseline tokens; outer admission required')
        for r in roster:
            if not profile.context_min <= r.cache_length + 1 <= r.cache_length + 1 + self.gamma <= profile.context_max:
                raise ValueError('Context outside frozen profile range; explicit replan required')
        frame = self._frames[epoch - 2] if epoch >= 2 else None
        if frame is not None and frame.profile.identity != profile.identity:
            raise ValueError('t-2 profile identity changed; explicit replan required')
        historical_k = search_capacity(frame) if frame is not None else len(roster)
        feasible = min(len(profile.sps), len(roster) + sum(min(self.gamma, r.remaining_output_budget - 1) for r in roster))
        reserved = max(len(roster), min(historical_k, feasible)) if roster else 0
        reasons = []
        if frame is None:
            reasons.append('cold_start_target_only_full_shadow')
        elif {(r.request, r.incarnation) for r in frame.roster} != keys:
            reasons.append('roster_churn')
        if reserved != historical_k:
            reasons.append('exogenous_feasibility_clamp')
        ticket = Capacity(epoch, session_epoch, roster, profile, frame.epoch if frame else None,
            frame.digest if frame else None, historical_k, reserved,
            profile.physical[reserved - 1] if reserved else 0, tuple(reasons), self.gamma, self._calibration)
        self._pending = ticket
        return ticket

    def _check(self, ticket):
        if ticket is not self._pending:
            raise ValueError('Stale or foreign capacity ticket')

    def allocate(self, ticket, conditional_scores):
        self._check(ticket)
        scores = _scores(conditional_scores, len(ticket.roster), self.gamma)
        lengths = [0] * len(ticket.roster)
        slots = ticket.reserved_k - len(ticket.roster)
        for score, _, _, position, i in _rank(ticket.roster, scores, budgeted=True):
            if slots == 0 or score == 0:
                break
            if position != lengths[i] + 1:
                raise RuntimeError('Non-prefix admission')
            lengths[i] = position
            slots -= 1
        b = len(ticket.roster) + sum(lengths)
        return Allocation(tuple(lengths), b, ticket.profile.physical[b - 1] if b else 0,
            ticket.reserved_k, _digest(scores))

    def bind(self, ticket, handles):
        """Bind observed scores to handles; private-source verification occurs later."""
        self._check(ticket)
        return BoundAllocation(self, ticket, handles)

    def finish(self, ticket, conditional_scores, *, shadow_positions, shadow_seconds):
        self._check(ticket)
        scores = _scores(conditional_scores, len(ticket.roster), self.gamma)
        _integer(shadow_positions, 'shadow_positions')
        if shadow_positions != len(ticket.roster) * self.gamma:
            raise ValueError('Full shadow draft work required even for zero admission')
        if isinstance(shadow_seconds, bool) or not isinstance(shadow_seconds, Real) or not math.isfinite(shadow_seconds) or shadow_seconds < 0 or (ticket.roster and shadow_seconds == 0):
            raise ValueError('Observed full shadow elapsed work must be charged')
        payload = dict(epoch=ticket.epoch, roster=[asdict(r) for r in ticket.roster],
            profile=ticket.profile.identity, calibration=ticket.calibration.identity, scores=scores, shadow_positions=shadow_positions,
            shadow_seconds=float(shadow_seconds))
        frame = History(ticket.epoch, ticket.roster, ticket.profile, scores, shadow_positions,
            float(shadow_seconds), _digest(payload))
        self._frames.append(frame)
        self._previous = {(r.request, r.incarnation) for r in ticket.roster}
        self._seen.update(self._previous)
        self._last_session_epoch = ticket.session_epoch
        self._pending = None
        return frame


_IDENTITY_FIELDS = ('request', 'incarnation', 'epoch', 'cache_length', 'nonce', 'proposal_limit', 'mode')


def _identity(handle):
    return {field: getattr(handle, field) for field in _IDENTITY_FIELDS}


class BoundAllocation:
    """Packed sampler typed hook; requires independently retained private logits.

    Observation logits select the proposed allocation. Validation recomputes from
    sampler-owned original logits and checks exact identities/context. This
    binds the numeric source; pre-token architecture and freeze-before-draw
    ordering still require source/caller review, not a self-declared string.
    """
    def __init__(self, planner, ticket, handles):
        self._planner, self._ticket = planner, ticket
        self._handles = dict(handles)
        if set(handles) != {r.request for r in ticket.roster}:
            raise ValueError('Complete current shadow handle roster required')
        self._identities = {name: _identity(h) for name, h in handles.items()}
        rows = []
        for r in ticket.roster:
            h = handles[r.request]
            if (h.request != r.request or h.incarnation != r.incarnation or h.epoch != ticket.session_epoch or
                    h.cache_length != r.cache_length or h.mode != 'shadow' or h.proposal_limit != ticket.gamma):
                raise ValueError('Handle does not match frozen pre-token scope/full shadow')
            logits = h.proposal.confidence_logits
            logits = logits.tolist() if hasattr(logits, 'tolist') else list(logits)
            if len(logits) != ticket.gamma:
                raise ValueError('Full confidence row required')
            rows.append(tuple(calibrated_probabilities(logits, ticket.calibration.temperatures)[0]))
        self.conditional_scores = tuple(rows)
        self.allocation = planner.allocate(ticket, self.conditional_scores)

    @property
    def allocations(self):
        return {r.request: length for r, length in zip(self._ticket.roster, self.allocation.lengths)}

    def validate_allocation(self, context, handles, allocations):
        ticket = self._ticket
        self._planner._check(ticket)
        if set(handles) != set(self._handles) or any(handles[r] is not h for r, h in self._handles.items()):
            raise ValueError('Foreign proposal handle')
        if context.get('epoch') != ticket.session_epoch:
            raise ValueError('Session epoch changed')
        roster = tuple(sorted(Request(**row) for row in context.get('roster', [])))
        if roster != ticket.roster:
            raise ValueError('Pre-token roster/cache/budget changed')
        if context.get('proposal_identity') != self._identities:
            raise ValueError('Private proposal identity mismatch')
        private_logits = context.get('proposal_confidence_logits', {})
        if set(private_logits) != set(self._handles):
            raise ValueError('Private confidence source required')
        rows = []
        for r in ticket.roster:
            logits = private_logits[r.request]
            if type(logits) is not tuple or len(logits) != ticket.gamma:
                raise ValueError('Immutable complete private confidence row required')
            rows.append(tuple(calibrated_probabilities(logits, ticket.calibration.temperatures)[0]))
        actual = self._planner.allocate(ticket, rows)
        if actual != self.allocation or allocations != self.allocations:
            raise ValueError('Allocation or confidence source changed')
        return dict(policy='two_step_capacity_cpu_reference_v1', planner_epoch=ticket.epoch,
            session_epoch=ticket.session_epoch, history_epoch=ticket.history_epoch,
            history_digest=ticket.history_digest, profile_identity=ticket.profile.identity,
            calibration_identity=ticket.calibration.identity, score_digest=actual.score_digest,
            proposal_identity=self._identities, historical_k=ticket.historical_k,
            reserved_k=ticket.reserved_k, reservation_physical_b=ticket.reservation_physical_b,
            logical_b=actual.logical_b, physical_b=actual.physical_b, replanning=list(ticket.replanning),
            full_shadow_required=True, hardware_overlap_proven=False)
