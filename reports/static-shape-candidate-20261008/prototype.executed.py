"""CPU-only fixed-query/StaticCache feasibility experiment, not production API.

Pinned to HF 5.17's StaticLayer.cumulative_length. StaticCache.crop is unsupported.
Both sequential and speculative paths use identical shared-bulk prefill, then
query width 8, key capacity C, explicit [1,1,8,C] masks and all-eight-row LM head.
The real GPU/BF16 failing prefixes are NOT tested by this CPU-only script.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path

import torch
from torch.nn import functional as F
import transformers
from transformers import Qwen3Config, Qwen3ForCausalLM, StaticCache


class FixedTarget:
    def __init__(self, model, capacity=64, width=8):
        self.model = model.eval().requires_grad_(False)
        self.capacity, self.width = capacity, width
        self.reset()

    def reset(self):
        self.cache = StaticCache(config=self.model.config, max_cache_len=self.capacity)
        self.length = 0

    def pointers(self):
        return [(layer.keys.data_ptr(), layer.values.data_ptr(), layer.cumulative_length.data_ptr())
                for layer in self.cache.layers if layer.is_initialized]

    @torch.no_grad()
    def crop(self, length):
        if type(length) is not int or not 0 <= length <= self.length:
            raise ValueError("Absolute retained prefix length out of range")
        before = self.pointers()
        for layer in self.cache.layers:
            if layer.is_initialized:
                # Pinned prototype adapter, NOT a supported HF StaticCache API.
                # Preserve tensor objects/storage addresses for future graphs.
                layer.cumulative_length.fill_(length)
        self.length = length
        assert self.pointers() == before

    @torch.no_grad()
    def forward(self, ids, valid):
        start, size = self.length, ids.shape[1]
        if ids.shape[0] != 1 or not 1 <= valid <= size or start + size > self.capacity:
            raise ValueError("Invalid query or insufficient physical cache headroom")
        positions = torch.arange(start, start + size, device=ids.device)[None]
        keys = torch.arange(self.capacity, device=ids.device)
        # Keep uncommitted/dummy suffix keys invisible even if stale values live
        # in physical backing storage. Real row i sees only positions <= start+i.
        visible = (keys[None, :] <= positions[0, :, None]) & (keys[None, :] < start + valid)
        dtype = self.model.dtype
        mask = torch.zeros((1, 1, size, self.capacity), device=ids.device, dtype=dtype)
        mask.masked_fill_(~visible[None, None], float("-inf"))
        before = self.pointers()
        result = self.model.model(input_ids=ids, position_ids=positions,
            attention_mask={"full_attention": mask}, past_key_values=self.cache,
            use_cache=True, return_dict=True)
        self.length = start + size
        assert all(int(layer.get_seq_length()) == self.length for layer in self.cache.layers)
        if before:
            assert self.pointers() == before
        # Deliberately project all rows for both baseline and verifier. Slicing
        # to only the first hidden row here would reintroduce M=1 versus M=8.
        return self.model.lm_head(result.last_hidden_state)

    @torch.no_grad()
    def prefill(self, prefix):
        self.reset()
        if prefix.shape[1]:
            return self.forward(prefix, prefix.shape[1])

    @torch.no_grad()
    def query(self, actual, dummy=0):
        if not 1 <= actual.shape[1] <= self.width:
            raise ValueError("Query must have between 1 and width real tokens")
        padded = actual.new_full((1, self.width), dummy)
        padded[:, :actual.shape[1]] = actual
        return self.forward(padded, actual.shape[1])

    @torch.no_grad()
    def greedy(self, prompt, count, eos=()):
        self.prefill(prompt[:, :-1])
        token = prompt[:, -1:]
        tokens, logits = [], []
        for _ in range(count):
            start = self.length
            prediction = self.query(token)[:, 0]
            logits.append(prediction.detach().clone())
            self.crop(start + 1)
            token = prediction.argmax(-1)[:, None]
            tokens.append(int(token.item()))
            if tokens[-1] in eos:
                break
        return tokens, logits


@contextmanager
def capture_shapes():
    original = F.scaled_dot_product_attention
    events = []
    def recording(q, k, v, attn_mask=None, *args, **kwargs):
        events.append((tuple(q.shape), tuple(k.shape), tuple(v.shape),
            tuple(attn_mask.shape) if attn_mask is not None else None,
            tuple(q.stride()), tuple(k.stride()),
            tuple(attn_mask.stride()) if attn_mask is not None else None))
        return original(q, k, v, attn_mask, *args, **kwargs)
    F.scaled_dot_product_attention = recording
    try:
        yield events
    finally:
        F.scaled_dot_product_attention = original


@torch.no_grad()
def oracle_spec(model, prompt, reference, reference_logits, count, rejection):
    fixed = FixedTarget(model)
    fixed.prefill(prompt[:, :-1])
    anchor = prompt[:, -1:]
    output, differences, rounds = [], [], []
    # First output uses exactly the same canonical shape and mask as baseline.
    before = fixed.length
    first = fixed.query(anchor)[:, 0].argmax(-1).item()
    fixed.crop(before + 1)
    output.append(first)
    while len(output) < count:
        offset, before = len(output), fixed.length
        proposals = reference[offset:offset + fixed.width - 1]
        if rejection is not None:
            proposals[rejection] = (proposals[rejection] + 1) % model.config.vocab_size
        verified = fixed.query(prompt.new_tensor([[output[-1]] + proposals]))[0]
        predictions = verified.argmax(-1).tolist()
        valid_rows = fixed.width if rejection is None else rejection + 1
        for row in range(valid_rows):
            # Compare only common-prefix positions; after the deliberately
            # wrong proposal, later rows have a different prefix by construction.
            expected = reference_logits[offset + row][0]
            differences.append({"row": row, "position": before + row,
                "max_logit_abs": float((verified[row].float() - expected.float()).abs().max()),
                "argmax_equal": int(verified[row].argmax()) == int(expected.argmax())})
        committed = []
        for proposal, prediction in zip(proposals, predictions):
            committed.append(prediction)
            if proposal != prediction:
                break
        else:
            committed.append(predictions[-1])
        committed = committed[:count - len(output)]
        output.extend(committed)
        fixed.crop(before + len(committed))
        assert fixed.length == prompt.shape[1] + len(output) - 1
        rounds.append({"before": before, "committed": len(committed), "after": fixed.length})
        # Preserve the first divergence instead of inventing common-prefix
        # comparisons after the two generated trajectories have diverged.
        if output != reference[:len(output)]:
            break
    return {"equal": output == reference[:count], "emitted": len(output),
        "max_common_prefix_logit_difference": max((d["max_logit_abs"] for d in differences), default=0),
        "common_prefix_argmax_mismatches": [d for d in differences if not d["argmax_equal"]],
        "rounds": len(rounds)}


@torch.no_grad()
def run(dtype):
    torch.manual_seed(79)
    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=256, attention_dropout=0.0)
    config._attn_implementation = "sdpa"
    model = Qwen3ForCausalLM(config).to(dtype=dtype).eval()
    prefix = torch.tensor([[1, 2, 3, 4]])
    target = FixedTarget(model)
    # Dummy suffix values may not influence any real output row.
    dummy_checks = []
    for valid in (1, 3, 8):
        actual = torch.arange(5, 5 + valid)[None]
        target.prefill(prefix)
        first = target.query(actual, dummy=0)[:, :valid].clone()
        target.prefill(prefix)
        second = target.query(actual, dummy=63)[:, :valid]
        assert torch.equal(first, second)
        dummy_checks.append(valid)
    # Deliberately poison a rejected suffix. Explicit valid/causal masks must
    # prevent its values from leaking into subsequent valid rows.
    poison_logits = []
    for poison in (-1000.0, 1000.0):
        target.prefill(prefix)
        target.query(torch.arange(5, 13)[None])
        target.crop(prefix.shape[1] + 1)
        for layer in target.cache.layers:
            layer.keys[:, :, target.length:].fill_(poison)
            layer.values[:, :, target.length:].fill_(poison)
        poison_logits.append(target.query(torch.tensor([[41, 42, 43]]))[:, :3].clone())
    assert torch.equal(*poison_logits)
    # Verify every retention point, including clearing all previously cached
    # tokens, keeps stable backing storage and the expected length.
    for keep in range(13):
        target.prefill(prefix)
        target.query(torch.arange(5, 13)[None])
        pointers = target.pointers()
        target.crop(keep)
        target.query(torch.tensor([[17]]))
        assert target.pointers() == pointers
        assert target.length == keep + 8
    # Prove that the supported library has no StaticLayer crop method.
    static_crop_supported = all(hasattr(layer, "crop") for layer in target.cache.layers)
    assert not static_crop_supported
    target.prefill(prefix)
    with capture_shapes() as shapes:
        target.query(torch.tensor([[9]]))
        target.crop(5)
        target.query(torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17]]))
        target.crop(8)
        target.query(torch.tensor([[18]]))
    assert len(set(shapes)) == 1
    # The LM head also always sees eight rows during decode/verify.
    head_shapes = []
    hook = model.lm_head.register_forward_pre_hook(lambda _m, inputs: head_shapes.append(tuple(inputs[0].shape)))
    cases = []
    try:
        for prompt_length in (1, 4, 17):
            prompt = torch.arange(1, prompt_length + 1)[None]
            reference, ref_logits = target.greedy(prompt, 32)
            for rejection in (None, 0, 3, 6):
                result = oracle_spec(model, prompt, reference, ref_logits, 23, rejection)
                cases.append({"prompt_length": prompt_length, "rejection": rejection, **result})
    finally:
        hook.remove()
    # Prefill uses the shared identical bulk-prefix route; decode/verify shapes
    # are independently recorded above and need not match the prefill length.
    return {"dtype": str(dtype), "dummy_suffix_noninterference_valid_lengths": dummy_checks,
        "stale_suffix_poison_noninterference": True, "rollback_positions_checked": 13,
        "static_layer_supports_crop": static_crop_supported,
        "decode_sdpa_shapes_strides": shapes[0], "unique_decode_sdpa_shapes": len(set(shapes)),
        "observed_lmhead_input_shapes_including_shared_prefill": sorted(set(head_shapes)),
        "cases": cases}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = {"scope": "CPU-only feasibility; no real GPU/BF16 failing-prefix gate or speedup claim",
        "torch": torch.__version__, "transformers": transformers.__version__,
        "query_width": 8, "key_capacity": 64,
        "prefill": "identical shared bulk prefix, excluding latest anchor, in both execution paths",
        "results": [run(torch.float32), run(torch.bfloat16)]}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({"output": str(output), "summary": [{"dtype": r["dtype"],
        "cases": len(r["cases"]), "equal_cases": sum(c["equal"] for c in r["cases"]),
        "max_logit_difference": max(c["max_common_prefix_logit_difference"] for c in r["cases"]),
        "unique_decode_shapes": r["unique_decode_sdpa_shapes"]} for r in report["results"]]}, indent=2))


if __name__ == "__main__":
    main()
