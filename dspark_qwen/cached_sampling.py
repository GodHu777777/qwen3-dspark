"""Stochastic cached reference with fixed admission, separate from greedy APIs.

At each completed round, both caches contain the committed sequence excluding
its latest emitted token. The target model keeps its configured forward dtype;
only probability arithmetic uses the explicit float64 tensor sampling law.
"""
from contextlib import nullcontext
from dataclasses import dataclass

import torch

from .cached_decode import DraftContextCache
from .tensor_sampling import (
    PROBABILITY_POLICY, TensorProposal, TensorRandom, TensorRound, logits_to_probabilities,
    sample_categorical, validate_stops, validate_temperature, verify_proposal,
)


@dataclass(frozen=True)
class SamplingObservation:
    """Synchronous view; observers must copy tensors they retain and not mutate."""
    proposal: TensorProposal
    target_probs: torch.Tensor
    result: TensorRound
    cache_before: int
    cache_after: int
    temperature: float
    probability_policy: str = PROBABILITY_POLICY

    @property
    def verified_proposal_length(self):
        return self.proposal.tokens.numel()

    @property
    def accepted_eos_position(self):
        if self.result.stop_reason == "eos" and self.result.extra_token_kind is None:
            return self.result.accepted_draft_tokens - 1
        return None


def _validate(target, input_ids, max_new_tokens, temperature, eos_ids):
    if type(max_new_tokens) is not int or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a nonnegative integer")
    target._validate_ids(input_ids)
    validate_temperature(temperature)
    return validate_stops(eos_ids, target.model.config.vocab_size)


@torch.no_grad()
def cached_target_sample(target, input_ids, max_new_tokens, *, temperature=1.0,
                         rng=None, eos_ids=()):
    """Target-only baseline using precisely the same probability adapter/draw.

    Equal seeds do not imply equal tokens versus speculation: random draws are
    consumed differently. Reproducibility means rerunning this same path/runtime.
    """
    eos_ids = _validate(target, input_ids, max_new_tokens, temperature, eos_ids)
    rng = TensorRandom() if rng is None else rng
    target.reset()
    if max_new_tokens == 0:
        return []
    output = []
    try:
        features = target.prefill(input_ids)
        while len(output) < max_new_tokens:
            law = logits_to_probabilities(target.predict(features, last_only=True), temperature)[0]
            token = sample_categorical(law, rng)
            output.append(int(token.item()))
            if output[-1] in eos_ids or len(output) == max_new_tokens:
                break
            features = target.append(token.reshape(1, 1))
        return output
    except Exception:
        target.reset()
        raise


@torch.no_grad()
def cached_speculative_sample(target, draft, input_ids, max_new_tokens, *,
                              temperature=1.0, rng=None, eos_ids=(), amp=False,
                              max_draft_tokens=None, observer=None,
                              trace_tokens=False, trace_probabilities=False):
    """Return token IDs and compact per-round records; full laws are opt-in.

    max_draft_tokens is fixed before random draws, capped by the remaining output
    budget. No confidence admission, STS calibration or scheduler is asserted.
    observer(SamplingObservation) sees actual q, p and pre-token raw confidence;
    it runs synchronously after cache commit and must not mutate the tensors.
    trace_probabilities stores detached CPU copies of full p/q in round records,
    with substantial memory/transfer cost. Neither trace option is for timing.

    Compatible with dynamic CachedTarget and the separate CanonicalTarget control
    through predict(features); it never changes either model's forward dtype.
    """
    eos_ids = _validate(target, input_ids, max_new_tokens, temperature, eos_ids)
    if tuple(target.layer_ids) != tuple(draft.spec.layer_ids):
        raise ValueError("Target feature layers must match draft layers")
    limit = draft.spec.block_size if max_draft_tokens is None else max_draft_tokens
    if type(limit) is not int or not 0 <= limit <= draft.spec.block_size:
        raise ValueError("max_draft_tokens must lie in [0, block_size]")
    rng = TensorRandom() if rng is None else rng
    target.reset()
    if max_new_tokens == 0:
        return [], []
    draft.eval()
    draft_cache = DraftContextCache(draft)
    autocast = lambda: torch.autocast("cuda", dtype=torch.bfloat16) if amp else nullcontext()
    output, rounds = [], []
    try:
        features = target.prefill(input_ids)
        first_law = logits_to_probabilities(target.predict(features, last_only=True), temperature)[0]
        first = int(sample_categorical(first_law, rng).item())
        output.append(first)
        if first in eos_ids or max_new_tokens == 1:
            return output, rounds
        with autocast():
            draft_cache.append(features.context)
        while len(output) < max_new_tokens:
            before = target.length
            if before != draft_cache.length or before != input_ids.shape[1] + len(output) - 1:
                raise RuntimeError("Committed target/draft cache invariant violated")
            remaining = max_new_tokens - len(output)
            anchor = input_ids.new_tensor([[output[-1]]])
            with autocast():
                proposal = draft.propose_stochastic_cached(anchor, draft_cache.layers, before,
                    temperature=temperature, rng=rng, max_draft_tokens=min(limit, remaining))
            verify_ids = torch.cat((anchor, proposal.tokens[None]), dim=1)
            verified = target.append(verify_ids)
            p = logits_to_probabilities(target.predict(verified)[0], temperature)
            decision = verify_proposal(proposal, p, rng, max_new_tokens=remaining, eos_ids=eos_ids)
            committed = decision.tokens.tolist()
            output.extend(committed)
            # Processed suffix includes OLD anchor, excludes NEW last output.
            keep = before + len(committed)
            target.crop(keep)
            with autocast():
                draft_cache.append(verified.context[:, :len(committed)])
            if target.length != keep or draft_cache.length != keep:
                raise RuntimeError("Rollback failed to restore committed-prefix caches")
            attempted = decision.accepted_draft_tokens + int(decision.rejected_index is not None)
            observation = SamplingObservation(proposal, p, decision, before, keep, float(temperature))
            record = dict(proposed=proposal.tokens.numel(),
                probability_policy=PROBABILITY_POLICY, temperature=float(temperature),
                accepted=decision.accepted_draft_tokens, attempted_positions=attempted,
                verified_proposal_length=observation.verified_proposal_length,
                accepted_eos_position=observation.accepted_eos_position,
                rejected_index=decision.rejected_index, extra_token_kind=decision.extra_token_kind,
                committed=len(committed), confidence_logits=proposal.confidence_logits.tolist(),
                # Diagnostic on PROPOSAL prefixes, including uncommitted tails.
                # STS uses realized prefix-event labels, never these overlaps.
                conditional_overlap=torch.minimum(p[:-1], proposal.draft_probs).sum(-1).tolist(),
                cache_before=before, verified_cache_end=verified.end,
                logical_verification_rows=verify_ids.shape[1],
                physical_verification_rows=getattr(verified, "padded_last", verified.last).shape[1],
                cache_after=keep, draft_cache_after=draft_cache.length,
                termination=decision.stop_reason if decision.stop_reason != "round_complete" else None)
            if trace_tokens:
                record.update(proposal_tokens=proposal.tokens.tolist(), committed_tokens=committed)
            if trace_probabilities:
                record.update(draft_probs=proposal.draft_probs.detach().cpu().clone(),
                              target_probs=p.detach().cpu().clone())
            rounds.append(record)
            if observer is not None:
                observer(observation)
            if decision.stop_reason == "eos":
                break
        return output, rounds
    except Exception:
        # Invalid laws, failed observers and failed projections cannot leave a
        # half-committed target state reusable by a later independent request.
        target.reset()
        draft_cache.reset()
        raise
