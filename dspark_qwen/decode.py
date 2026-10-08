"""Correctness-first greedy draft/verify. Full-prefix recomputation; not a serving engine.

No KV cache, random sampling, dynamic prefix scheduler, or speedup claims.
"""
from contextlib import nullcontext
import torch


@torch.no_grad()
def target_greedy(target, input_ids, max_new_tokens, eos_ids):
    ids = input_ids.clone()
    output = []
    for _ in range(max_new_tokens):
        features = target.capture(ids)
        token = int(target.logits(features.last[:, -1]).argmax(-1))
        output.append(token)
        ids = torch.cat((ids, ids.new_tensor([[token]])), dim=1)
        if token in eos_ids:
            break
    return output


@torch.no_grad()
def speculative_greedy(target, draft, input_ids, max_new_tokens, eos_ids, amp=False):
    if max_new_tokens < 1:
        return [], []
    draft.eval()
    first = target_greedy(target, input_ids, 1, eos_ids)[0]
    output, rounds = [first], []
    ids = torch.cat((input_ids, input_ids.new_tensor([[first]])), dim=1)
    if first in eos_ids:
        return output, rounds
    while len(output) < max_new_tokens:
        # The latest anchor has been sampled but not processed by target yet.
        context = target.capture(ids[:, :-1]).context
        with torch.autocast("cuda", dtype=torch.bfloat16) if amp else nullcontext():
            proposals, confidence = draft.propose_greedy(ids, context)
        proposal_ids = ids.new_tensor([proposals])
        verified = target.capture(torch.cat((ids, proposal_ids), dim=1))
        predictions = target.logits(verified.last[:, ids.shape[1] - 1:]).argmax(-1)[0].tolist()
        accepted, committed = 0, []
        for candidate, correct in zip(proposals, predictions):
            if candidate != correct:
                committed.append(correct)  # First rejection: target correction, discard suffix.
                break
            committed.append(candidate)
            accepted += 1
            if candidate in eos_ids:
                break
        else:
            committed.append(predictions[-1])  # All accepted: one target bonus token.
        rounds.append({"proposed": len(proposals), "accepted": accepted,
                       "confidence": confidence})
        for token in committed:
            output.append(token)
            ids = torch.cat((ids, ids.new_tensor([[token]])), dim=1)
            if token in eos_ids or len(output) >= max_new_tokens:
                return output, rounds
    return output, rounds
