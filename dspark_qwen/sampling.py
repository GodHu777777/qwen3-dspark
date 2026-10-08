"""Small-vocabulary CPU reference for stochastic speculative verification.

Consumes probabilities, not logits. No model, cache, scheduler or tensor backend
is involved. See docs/stochastic-sampling.md for the distribution/admission
contract and the distinction between mathematical exactness and floating point.
"""
from dataclasses import dataclass
import math
from numbers import Real
from typing import Callable, Sequence


Distribution = tuple[float, ...]
Uniform = Callable[[], float]


def probabilities(values: Sequence[float]) -> Distribution:
    """Validate a probability vector; renormalize only <=1e-12 sum roundoff.

    Unnormalized weights/logits, negative values and non-finite values are errors.
    This tolerance is solely an input sum check, never an acceptance epsilon.
    """
    row = tuple(values)
    if not row:
        raise ValueError("Probability vector must be nonempty")
    if any(isinstance(x, bool) or not isinstance(x, Real) or
           not 0 <= x <= 1 or not math.isfinite(x) for x in row):
        raise ValueError("Probabilities must be finite real numbers in [0, 1]")
    total = math.fsum(row)
    if abs(total - 1.0) > 1e-12:
        raise ValueError("Probabilities must sum to one (absolute tolerance 1e-12)")
    return tuple(float(x) / total for x in row)


def _uniform(rng: Uniform) -> float:
    value = rng()
    if (isinstance(value, bool) or not isinstance(value, Real) or
            not 0 <= value < 1 or not math.isfinite(value)):
        raise ValueError("Random source must return a finite value in [0, 1)")
    return float(value)


def _categorical(row: Distribution, rng: Uniform) -> int:
    u = _uniform(rng)
    cumulative = 0.0
    last_positive = 0
    for token, mass in enumerate(row):
        if mass > 0:
            last_positive = token
            cumulative += mass
            if u < cumulative:
                return token
    # Summation roundoff may leave a tiny gap below 1. Never select zero support.
    return last_positive


def sample_categorical(values: Sequence[float], rng: Uniform) -> int:
    return _categorical(probabilities(values), rng)


def _pair(target, draft):
    p, q = probabilities(target), probabilities(draft)
    if len(p) != len(q):
        raise ValueError("Target and draft vocabulary sizes must match")
    return p, q


def _token(token, vocab):
    if type(token) is not int or not 0 <= token < vocab:
        raise ValueError("Token ID must be an integer in the vocabulary")


def acceptance_probability(target, draft, token: int) -> float:
    """min(1, p[token]/q[token]); a q-zero proposal is an invalid event."""
    p, q = _pair(target, draft)
    _token(token, len(p))
    if q[token] == 0:
        raise ValueError("Proposed token has zero probability under its draft law")
    return min(1.0, p[token] / q[token])


def _residual(p, q):
    positive = tuple(max(pi - qi, 0.0) for pi, qi in zip(p, q))
    mass = math.fsum(positive)
    if mass == 0:
        raise ValueError("Zero residual mass: rejection is impossible when p=q")
    return tuple(value / mass for value in positive)


def residual_distribution(target, draft) -> Distribution:
    """Normalize (p-q)+ without a small-mass threshold or target fallback."""
    return _residual(*_pair(target, draft))


@dataclass(frozen=True)
class Proposal:
    tokens: tuple[int, ...]
    draft_probs: tuple[Distribution, ...]


@dataclass(frozen=True)
class SampledRound:
    tokens: tuple[int, ...]  # New tokens only; excludes any pre-existing anchor.
    accepted_draft_tokens: int
    rejected_index: int | None  # Zero-based; None if no rejection was reached.
    extra_token_kind: str | None  # residual, bonus, or None (EOS/budget).
    stop_reason: str  # eos, budget, or round_complete


def _budget(value, label):
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")


def _stops(stop_token_ids, vocab):
    stops = tuple(stop_token_ids)
    for token in stops:
        _token(token, vocab)
    return frozenset(stops)


def sample_proposal(draft_distribution: Callable[[tuple[int, ...]], Sequence[float]],
                    max_draft_tokens: int, rng: Uniform, *,
                    admit: Callable[[tuple[int, ...]], bool] | None = None) -> Proposal:
    """Sample conditional q rows, keeping exactly the laws used for each draw.

    The callback sees the already sampled block prefix. Admission runs BEFORE
    sampling this token; it may depend on this prefix, never this token or future
    random draws. The caller owns any closed-over model state and this causality
    contract. Do not truncate proposals by inspecting the candidate being admitted
    (including EOS). Verification handles EOS after acceptance/correction.
    """
    _budget(max_draft_tokens, "max_draft_tokens")
    tokens, rows = [], []
    for _ in range(max_draft_tokens):
        prefix = tuple(tokens)
        if admit is not None:
            decision = admit(prefix)
            if type(decision) is not bool:
                raise ValueError("Admission callback must return bool")
            if not decision:
                break
        row = probabilities(draft_distribution(prefix))
        if rows and len(row) != len(rows[0]):
            raise ValueError("Draft vocabulary size changed within a proposal")
        tokens.append(_categorical(row, rng))
        rows.append(row)
    return Proposal(tuple(tokens), tuple(rows))


def verify_proposal(proposal: Proposal, target_probs: Sequence[Sequence[float]],
                    rng: Uniform, *, max_new_tokens: int,
                    stop_token_ids: Sequence[int] = ()) -> SampledRound:
    """Verify one prefix, then emit first rejection residual or all-accept bonus.

    n draft rows require n+1 target rows. Row j predicts the token AFTER the
    committed context plus proposal.tokens[:j]; the last row is the bonus law.
    All inputs (including unused tails) are checked before consuming randomness.
    max_new_tokens is the remaining output budget, excluding an existing anchor.
    EOS is included, with no subsequent draw; budget zero consumes no randomness.
    Neither the actual provenance of q nor target prefix alignment is inferable
    from these arrays: both remain explicit caller responsibilities.
    """
    _budget(max_new_tokens, "max_new_tokens")
    tokens = tuple(proposal.tokens)
    q = tuple(probabilities(row) for row in proposal.draft_probs)
    p = tuple(probabilities(row) for row in target_probs)
    if len(q) != len(tokens) or len(p) != len(tokens) + 1:
        raise ValueError("n proposed tokens require n draft and n+1 target rows")
    vocab = len(p[0])
    if any(len(row) != vocab for row in p + q):
        raise ValueError("Target and draft vocabulary sizes must match")
    stops = _stops(stop_token_ids, vocab)
    for token, row in zip(tokens, q):
        _token(token, vocab)
        if row[token] == 0:
            raise ValueError("Proposed token has zero probability under its draft law")
    if max_new_tokens == 0:
        return SampledRound((), 0, None, None, "budget")
    emitted = []
    for j, token in enumerate(tokens):
        alpha = min(1.0, p[j][token] / q[j][token])
        if _uniform(rng) >= alpha:
            replacement = _categorical(_residual(p[j], q[j]), rng)
            emitted.append(replacement)
            reason = "eos" if replacement in stops else (
                "budget" if len(emitted) == max_new_tokens else "round_complete")
            return SampledRound(tuple(emitted), j, j, "residual", reason)
        emitted.append(token)
        if token in stops or len(emitted) == max_new_tokens:
            return SampledRound(tuple(emitted), j + 1, None, None,
                                "eos" if token in stops else "budget")
    bonus = _categorical(p[-1], rng)
    emitted.append(bonus)
    reason = "eos" if bonus in stops else (
        "budget" if len(emitted) == max_new_tokens else "round_complete")
    return SampledRound(tuple(emitted), len(tokens), None, "bonus", reason)
