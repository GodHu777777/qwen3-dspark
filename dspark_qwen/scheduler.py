"""CPU reference for DSpark paper v1 Algorithm 1, across active requests.

Source: https://arxiv.org/html/2607.05147v1#alg1 (2026-07-06).
Inputs are already calibrated CONDITIONAL acceptance probabilities. The caller
must supply pre-token, non-anticipating scores; this pure planner cannot verify
their provenance. SPS is measured engine forward steps/second for logical target
verification batch size B, not tokens/second or a single-request draft-length
cost. If the engine pads/buckets B, its actual physical execution cost must be
included in SPS(B), with physical batch sizes recorded by the profiling caller.

This implements the paper's synchronous first-non-improvement break, not an
unconstrained global retrospective search. It need not find the global maximum
on a jagged capacity curve. No GPU execution, STS, two-step asynchronous state,
kernel packing, engine integration, or performance benefit is implemented here.
"""
from dataclasses import dataclass
import math
from numbers import Real
from typing import Mapping, Sequence


@dataclass(frozen=True)
class Admission:
    request: int
    position: int  # one-based draft position
    prefix_survival: float
    trial_batch_tokens: int
    trial_expected_tokens: float
    trial_tokens_per_second: float
    admitted: bool


@dataclass(frozen=True)
class PrefixPlan:
    prefix_lengths: tuple[int, ...]
    target_batch_tokens: int
    expected_tokens: float  # Includes one baseline token per active request.
    estimated_tokens_per_second: float
    stopping_reason: str
    admissions: tuple[Admission, ...]


def _finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite real number")
    return float(value)


def plan_prefixes(confidences: Sequence[Sequence[float]], sps: Mapping[int, float],
                  *, max_batch_tokens: int | None = None) -> PrefixPlan:
    """Allocate one shared target token budget over R requests.

    All rows must have the same maximum draft length gamma, as in Algorithm 1.
    Every integer SPS(B) from R through the feasible upper bound is required:
    no interpolation or smoothing across hardware capacity cliffs. Optional
    max_batch_tokens is an explicit engine capacity bound, not a per-request k.
    If it cannot fit R baseline tokens, request admission belongs to an outer
    serving scheduler and this function refuses to silently drop requests.

    R=0 is an empty/no-forward plan. gamma=0 is valid target-only operation.
    Probability ties are ordered by request then position so each selected
    request remains a prefix even when confidences equal one.
    """
    rows = [tuple(row) for row in confidences]
    requests = len(rows)
    if max_batch_tokens is not None and (type(max_batch_tokens) is not int or max_batch_tokens < 0):
        raise ValueError("max_batch_tokens must be a nonnegative integer")
    if requests == 0:
        return PrefixPlan((), 0, 0.0, 0.0, "no_active_requests", ())
    gamma = len(rows[0])
    if any(len(row) != gamma for row in rows):
        raise ValueError("Every request must have the same maximum draft length")
    maximum = requests * (gamma + 1)
    budget = maximum if max_batch_tokens is None else min(max_batch_tokens, maximum)
    if budget < requests:
        raise ValueError("Target budget cannot fit one baseline token per active request")
    curve = {}
    for batch in range(requests, budget + 1):
        if batch not in sps:
            raise ValueError(f"Missing discrete SPS({batch}); interpolation is not allowed")
        value = _finite_number(sps[batch], f"SPS({batch})")
        if value <= 0:
            raise ValueError("SPS must be positive; represent unsupported shapes with a capacity bound")
        curve[batch] = value
    candidates = []
    for request, row in enumerate(rows):
        survival = 1.0
        for position, value in enumerate(row, start=1):
            confidence = _finite_number(value, "confidence")
            if not 0 <= confidence <= 1:
                raise ValueError("Conditional confidence must lie in [0, 1]")
            survival *= confidence
            # Paper Algorithm 1, line 4 explicitly uses a[r,j] > 0. Retain
            # that filter even if an unusual rising SPS curve could reward
            # admitting zero-mass tokens; this is not unrestricted global search.
            if survival > 0:
                candidates.append((survival, request, position))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    lengths = [0] * requests
    batch, expected = requests, float(requests)
    score = expected * curve[batch]
    if not math.isfinite(score):
        raise ValueError("Throughput objective overflow")
    history = []
    reason = "all_positive_extensions_considered"
    for survival, request, position in candidates:
        if batch == budget:
            reason = "target_batch_capacity"
            break
        if position != lengths[request] + 1:
            raise RuntimeError("Sorted candidate violated the prefix dependency")
        trial_expected = expected + survival
        trial_score = trial_expected * curve[batch + 1]
        if not math.isfinite(trial_score):
            raise ValueError("Throughput objective overflow")
        improve = trial_score > score  # No epsilon or look-ahead beyond first drop.
        history.append(Admission(request, position, survival, batch + 1,
            trial_expected, trial_score, improve))
        if not improve:
            reason = "first_non_improvement"
            break
        lengths[request] = position
        batch, expected, score = batch + 1, trial_expected, trial_score
    return PrefixPlan(tuple(lengths), batch, expected, score, reason, tuple(history))
