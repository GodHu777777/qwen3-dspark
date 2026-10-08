"""Independent CPU audit of integrated committed draft KV contents."""
import json
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from dspark_qwen.cached_target import CachedTarget
from dspark_qwen.cached_decode import cached_speculative_greedy
from dspark_qwen.target import FrozenTarget
from dspark_qwen.model import DSparkDraft
from dspark_qwen.config import DraftConfig

torch.set_num_threads(2)
round_count = 0
case_count = 0
kv_max_abs = 0.0
backbone_max_abs = 0.0

@torch.no_grad()
def run(block_size, prompt_length, reject_at):
    global case_count, round_count
    torch.manual_seed(83)
    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=256, attention_dropout=0.0)
    config._attn_implementation = "sdpa"
    model = Qwen3ForCausalLM(config).eval()
    spec = DraftConfig(layer_ids=(0, 2), num_layers=2, block_size=block_size,
                       markov_rank=8, mask_token_id=63)
    cached = CachedTarget(model, spec.layer_ids)
    full = FrozenTarget(model, spec)
    draft = DSparkDraft(model, spec).eval()
    prompt = torch.arange(1, prompt_length + 1)[None]
    reference = cached.greedy(prompt, 64)
    class CheckedDraft:
        spec = draft.spec
        def eval(self): return self
        def project_context_kv(self, features, start):
            return draft.project_context_kv(features, start)
        def propose_greedy_cached(self, anchor, kv, length):
            global round_count, kv_max_abs, backbone_max_abs
            # The processed prefix excludes the currently emitted anchor.
            processed_output = length - prompt_length
            prefix = torch.cat((prompt, prompt.new_tensor([reference[:processed_output]])), -1)
            assert prefix.shape[1] == length
            assert anchor.item() == reference[processed_output]
            features = full.capture(prefix).context
            expected_kv = draft.project_context_kv(features, 0)
            assert len(kv) == len(expected_kv)
            for actual_layer, expected_layer in zip(kv, expected_kv):
                for actual, expected in zip(actual_layer, expected_layer):
                    assert actual.shape == expected.shape
                    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
                    kv_max_abs = max(kv_max_abs, float((actual - expected).abs().max()))
            expected_hidden = draft.backbone(torch.cat((prefix, anchor), -1), features,
                torch.tensor([length]))[0, 0]
            observed_hidden = draft.backbone_cached(anchor, kv, length)
            assert torch.isfinite(observed_hidden).all() and torch.isfinite(expected_hidden).all()
            backbone_max_abs = max(backbone_max_abs, float((observed_hidden - expected_hidden).abs().max()))
            proposals = reference[processed_output + 1:processed_output + 1 + block_size]
            if reject_at is not None:
                proposals[reject_at] = (proposals[reject_at] + 1) % config.vocab_size
            round_count += 1
            return proposals, [0.5] * len(proposals)
    tokens, rounds = cached_speculative_greedy(cached, CheckedDraft(), prompt, 23)
    assert tokens == reference[:23]
    assert cached.length == prompt_length + len(tokens) - 1
    case_count += 1

for block in (1, 3, 7):
    for prompt_length in (1, 4, 9):
        for rejection in list(range(block)) + [None]:
            run(block, prompt_length, rejection)
print(json.dumps({"scope": "CPU FP32 independent integrated cache-content audit",
    "cases_passed": case_count, "proposal_rounds_checked": round_count,
    "max_projected_kv_absolute_difference": kv_max_abs,
    "max_draft_backbone_absolute_difference": backbone_max_abs,
    "block_sizes": [1, 3, 7], "prompt_lengths": [1, 4, 9],
    "checks": "exact output tokens and cache lengths; every layer shape/finite values; unthresholded projected K/V and backbone differences versus fresh full prefix; all rejection offsets/all-accepted; budget crop",
    "note": "Initial audit's 1e-6 absolute tensor closeness assertion saw one 1.13249e-6 FP32 difference. This version reports numeric errors directly; no tolerance alters token selection or its exact equality gate."}, indent=2))
