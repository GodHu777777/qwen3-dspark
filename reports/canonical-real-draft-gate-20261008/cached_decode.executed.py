"""Greedy speculative decoding with target and projected draft context caches.

At every externally visible boundary BOTH caches contain exactly the committed
prefix excluding the latest emitted anchor. Verification may extend target KV
past a rejection; crop it before the next proposal. Only committed features are
projected into draft KV, so rejected features never enter the draft cache.

Block verification and sequential target decoding can pick different BF16
argmaxes near ties. Equality is an empirical check for a given dtype/backend,
not a cross-kernel numerical guarantee. CPU FP32 tests exercise logical parity.
"""
from contextlib import nullcontext

import torch


class DraftContextCache:
    def __init__(self, draft):
        self.draft = draft
        self.reset()

    def reset(self):
        self.layers = []
        self.length = 0

    @torch.no_grad()
    def append(self, features):
        if features is None or features.ndim != 3 or features.shape[0] != 1 or features.shape[1] < 1:
            raise ValueError("Expected one nonempty target feature chunk")
        projected = self.draft.project_context_kv(features, self.length)
        if self.layers:
            projected = [(torch.cat((old_k, k), dim=2), torch.cat((old_v, v), dim=2))
                for (old_k, old_v), (k, v) in zip(self.layers, projected)]
        self.layers = projected
        self.length += features.shape[1]

    def crop(self, length):
        if type(length) is not int or not 0 <= length <= self.length:
            raise ValueError("Invalid draft cache crop length")
        self.layers = [(k[:, :, :length], v[:, :, :length]) for k, v in self.layers]
        self.length = length


@torch.no_grad()
def cached_speculative_greedy(target, draft, input_ids, max_new_tokens, eos_ids=(), amp=False, trace_tokens=False):
    """Return tokens and auditable per-round cache/commit counts.

Target must be a CachedTarget configured with draft.spec.layer_ids. After return,
target.length == prompt length + emitted count - 1 (unless count is zero).
The latest output token, including EOS, is intentionally left unprocessed.
"""
    if type(max_new_tokens) is not int or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a nonnegative integer")
    target._validate_ids(input_ids)
    if tuple(target.layer_ids) != tuple(draft.spec.layer_ids):
        raise ValueError("Target feature layers must match draft layers")
    target.reset()
    if max_new_tokens == 0:
        return [], []
    draft.eval()
    eos_ids = set(eos_ids)
    features = target.prefill(input_ids)
    first = int(target.predict(features, last_only=True).argmax(-1).item())
    output, rounds = [first], []
    if first in eos_ids or max_new_tokens == 1:
        return output, rounds
    draft_cache = DraftContextCache(draft)
    autocast = lambda: torch.autocast("cuda", dtype=torch.bfloat16) if amp else nullcontext()
    with autocast():
        draft_cache.append(features.context)
    while len(output) < max_new_tokens:
        before = target.length
        if before != draft_cache.length or before != input_ids.shape[1] + len(output) - 1:
            raise RuntimeError("Committed target/draft cache invariant violated")
        anchor = input_ids.new_tensor([[output[-1]]])
        with autocast():
            proposals, confidence = draft.propose_greedy_cached(anchor, draft_cache.layers, before)
        if not proposals:
            raise ValueError("Draft returned an empty proposal")
        # Row 0 predicts proposal[0]; final row predicts the all-accepted bonus.
        verified = target.append(input_ids.new_tensor([[output[-1]] + proposals]))
        verified_logits = target.predict(verified)
        predictions = verified_logits.argmax(-1)[0].tolist()
        accepted, committed = 0, []
        for proposed, correct in zip(proposals, predictions):
            if proposed != correct:
                committed.append(correct)
                break
            committed.append(proposed)
            accepted += 1
            if proposed in eos_ids:
                break
        else:
            committed.append(predictions[-1])
        eos_position = next((i for i, token in enumerate(committed) if token in eos_ids), None)
        if eos_position is not None:
            committed = committed[:eos_position + 1]
        committed = committed[:max_new_tokens - len(output)]
        output.extend(committed)
        # Includes old anchor and excludes newest output token, even when the
        # last output is an accepted draft token at an EOS/budget boundary.
        keep = before + len(committed)
        target.crop(keep)
        with autocast():
            draft_cache.append(verified.context[:, :len(committed)])
        if draft_cache.length != keep or target.length != keep:
            raise RuntimeError("Rollback failed to restore committed-prefix caches")
        rounds.append({"proposed": len(proposals), "accepted": accepted,
            "committed_accepted": min(accepted, len(committed)),
            "committed": len(committed), "confidence": confidence,
            "cache_before": before, "verified_length": verified.end,
            "cache_after": keep, "draft_cache_after": draft_cache.length,
            "termination": "eos" if output[-1] in eos_ids else
                "max_new_tokens" if len(output) == max_new_tokens else None})
        if trace_tokens:
            values, indices = verified_logits.float().topk(min(5, verified_logits.shape[-1]), dim=-1)
            rounds[-1].update(proposal_tokens=proposals, verified_predictions=predictions,
                committed_tokens=committed, verified_top_values=values[0].tolist(),
                verified_top_tokens=indices[0].tolist())
        if output[-1] in eos_ids:
            break
    return output, rounds
