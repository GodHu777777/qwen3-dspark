"""Tensor probability reference; float64 law, positive temperature, no filtering.

The model's forward dtype is independent of this probability dtype. Normalizing
once before sampling defines a local discrete law, not cross-kernel equality.
This readable implementation synchronizes validation/acceptance scalars; it is
not an optimized GPU sampler. No denominator epsilon or residual fallback.
"""
from dataclasses import dataclass
import math
from numbers import Real

import torch


PROBABILITY_POLICY = "float64_softmax_normalize_cdf_v1"


def validate_temperature(temperature):
    if (isinstance(temperature, bool) or not isinstance(temperature, Real) or
            not math.isfinite(temperature) or temperature <= 0):
        raise ValueError("temperature must be finite and strictly positive")
    return float(temperature)


def check_probabilities(probs):
    """Check an already defined float64 law without silently renormalizing it."""
    if (not isinstance(probs, torch.Tensor) or probs.dtype != torch.float64 or
            probs.ndim < 1 or probs.shape[-1] < 1):
        raise ValueError("Expected float64 probability tensor with a nonempty vocabulary")
    if (not bool(torch.isfinite(probs).all()) or bool((probs < 0).any()) or
            bool((probs > 1).any()) or
            bool(((probs.sum(-1) - 1).abs() > 1e-12).any())):
        raise ValueError("Probabilities must be finite, in [0,1], and sum to one within 1e-12")
    return probs


def logits_to_probabilities(logits, temperature):
    """Convert finite logits to the actual sampling law used by both p and q."""
    temperature = validate_temperature(temperature)
    if (not isinstance(logits, torch.Tensor) or not logits.is_floating_point() or
            logits.ndim < 1 or logits.shape[-1] < 1 or not bool(torch.isfinite(logits).all())):
        raise ValueError("Expected finite floating logits and a nonempty vocabulary")
    scaled = logits.to(torch.float64) / temperature
    if not bool(torch.isfinite(scaled).all()):
        raise ValueError("Temperature scaling overflowed")
    weights = scaled.softmax(-1)
    law = weights / weights.sum(-1, keepdim=True)
    return check_probabilities(law)


class TensorRandom:
    """Dedicated generator recommended for seeded same-runtime reproducibility."""
    def __init__(self, generator=None):
        self.generator = generator

    def uniform(self, reference):
        return torch.rand((), dtype=torch.float64, device=reference.device, generator=self.generator)


def _uniform(rng, reference):
    value = rng.uniform(reference)
    if (not isinstance(value, torch.Tensor) or value.shape != () or
            value.dtype != torch.float64 or value.device != reference.device or
            not bool(torch.isfinite(value)) or not bool((value >= 0) & (value < 1))):
        raise ValueError("RNG must yield a float64 scalar in [0,1) on the probability device")
    return value


def sample_categorical(probs, rng):
    """One token as a scalar long tensor; strict CDF boundary, zero support skipped."""
    check_probabilities(probs)
    if probs.ndim != 1:
        raise ValueError("Categorical sampling expects one row")
    index = torch.searchsorted(probs.cumsum(-1), _uniform(rng, probs), right=True)
    last_positive = torch.nonzero(probs > 0, as_tuple=False)[-1, 0]
    return torch.where(index < probs.numel(), index, last_positive)


def residual_distribution(target, draft):
    check_probabilities(target)
    check_probabilities(draft)
    if target.ndim != 1 or target.shape != draft.shape or target.device != draft.device:
        raise ValueError("Residual requires matching single-row target/draft laws")
    positive = (target - draft).relu()
    mass = positive.sum()
    if not bool(mass > 0):
        raise ValueError("Zero residual mass: rejection is impossible for identical laws")
    return check_probabilities(positive / mass)


@dataclass(frozen=True)
class TensorProposal:
    tokens: torch.Tensor  # [n], long
    draft_probs: torch.Tensor  # [n,V], float64; the actual sampled q rows
    confidence_logits: torch.Tensor  # [n], pre-current-token raw head outputs


@dataclass(frozen=True)
class TensorRound:
    tokens: torch.Tensor
    accepted_draft_tokens: int
    rejected_index: int | None
    extra_token_kind: str | None
    stop_reason: str


def validate_stops(eos_ids, vocab):
    values = tuple(eos_ids)
    if any(type(token) is not int or not 0 <= token < vocab for token in values):
        raise ValueError("EOS IDs must be integers in the vocabulary")
    return frozenset(values)


def verify_proposal(proposal, target_probs, rng, *, max_new_tokens, eos_ids=()):
    """n proposals + n+1 target rows; return only newly committed output tokens."""
    if type(max_new_tokens) is not int or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a nonnegative integer")
    tokens, q, p = proposal.tokens, proposal.draft_probs, target_probs
    check_probabilities(q)
    check_probabilities(p)
    if (not isinstance(tokens, torch.Tensor) or tokens.dtype != torch.long or tokens.ndim != 1 or
            q.ndim != 2 or p.ndim != 2 or q.shape[0] != tokens.numel() or
            p.shape != (tokens.numel() + 1, q.shape[1]) or
            tokens.device != q.device or p.device != q.device):
        raise ValueError("Require [n] long tokens, [n,V] q and [n+1,V] p on one device")
    confidence = proposal.confidence_logits
    if (not isinstance(confidence, torch.Tensor) or confidence.shape != tokens.shape or
            not confidence.is_floating_point() or confidence.device != q.device or
            not bool(torch.isfinite(confidence).all())):
        raise ValueError("Expected finite [n] confidence logits on the probability device")
    if bool(((tokens < 0) | (tokens >= q.shape[-1])).any()):
        raise ValueError("Proposal token outside vocabulary")
    selected_q = q.gather(-1, tokens[:, None]).flatten()
    if bool((selected_q == 0).any()):
        raise ValueError("Proposal token has zero mass under its actual sampling law")
    stops = validate_stops(eos_ids, q.shape[-1])
    emitted = []

    def result(accepted, rejected, extra, reason):
        values = torch.stack(emitted) if emitted else tokens.new_empty((0,))
        return TensorRound(values, accepted, rejected, extra, reason)

    def reason(token):
        if int(token.item()) in stops:
            return "eos"
        return "budget" if len(emitted) == max_new_tokens else "round_complete"

    if max_new_tokens == 0:
        return result(0, None, None, "budget")
    for j, token in enumerate(tokens):
        alpha = torch.minimum(p[j, token] / selected_q[j], p.new_ones(()))
        if bool(_uniform(rng, p) >= alpha):
            replacement = sample_categorical(residual_distribution(p[j], q[j]), rng)
            emitted.append(replacement)
            return result(j, j, "residual", reason(replacement))
        emitted.append(token)
        stop = reason(token)
        if stop != "round_complete":
            return result(j + 1, None, None, stop)
    bonus = sample_categorical(p[-1], rng)
    emitted.append(bonus)
    return result(tokens.numel(), None, "bonus", reason(bonus))
