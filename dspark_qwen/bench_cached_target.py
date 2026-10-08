"""Cached target-only timing and isolated target verification-block costs.

No draft is executed: these measurements are NOT speculative serving speedups.
Example: python -m dspark_qwen.bench_cached_target --model /path/to/Qwen3-0.6B
  --records /path/to/records.jsonl --output /new/path/cached-target.json
CPU harness check: --tiny --device cpu --warmup 1 --repeats 2 --max-new-tokens 8
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, GenerationConfig, Qwen3Config, Qwen3ForCausalLM

from .cached_target import CachedTarget


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(device, fn):
    synchronize(device)
    start = time.perf_counter()
    value = fn()
    synchronize(device)
    return value, (time.perf_counter() - start) * 1000


def distribution(values):
    ordered = sorted(values)
    def percentile(q):
        offset = q * (len(ordered) - 1)
        lo = int(offset)
        hi = min(lo + 1, len(ordered) - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (offset - lo)
    return {"count": len(values), "median": statistics.median(values),
            "p95": percentile(0.95), "min": min(values), "max": max(values)}


@torch.no_grad()
def generation_trial(target, ids, count, eos):
    device = ids.device
    def prefill():
        features = target.prefill(ids)
        return target.logits(features.last[:, -1]).argmax(-1)
    synchronize(device)
    begin = time.perf_counter()
    first, prefill_ms = timed(device, prefill)
    tokens = [int(first.item())]
    def decode():
        token = first
        while len(tokens) < count and tokens[-1] not in eos:
            features = target.append(token[:, None])
            token = target.logits(features.last[:, -1]).argmax(-1)
            tokens.append(int(token.item()))
    _, decode_ms = timed(device, decode)
    synchronize(device)
    total_ms = (time.perf_counter() - begin) * 1000
    return {"tokens": tokens, "prefill_first_logit_ms": prefill_ms,
            "decode_remaining_ms": decode_ms, "e2e_ms": total_ms,
            "output_tokens": len(tokens), "decode_steps": len(tokens) - 1,
            "ended_eos": tokens[-1] in eos}


@torch.no_grad()
def check_recompute(target, ids, count, eos):
    cached = target.greedy(ids, count, eos)
    full = []
    prefix = ids.clone()
    for _ in range(count):
        hidden = target.model.model(prefix, use_cache=False, return_dict=True).last_hidden_state[:, -1]
        token = target.logits(hidden).argmax(-1)
        full.append(int(token.item()))
        if full[-1] in eos:
            break
        prefix = torch.cat((prefix, token[:, None]), -1)
    return {"cached_tokens": cached, "full_recompute_tokens": full, "equal": cached == full}


@torch.no_grad()
def check_hf_cached(target, ids, count, eos):
    """Compare to HF's same-dtype cached greedy policy, with no extra processors."""
    cached = target.greedy(ids, count, eos)
    config = GenerationConfig(do_sample=False, max_new_tokens=count, use_cache=True,
        cache_implementation="dynamic", eos_token_id=sorted(eos) if eos else None,
        pad_token_id=target.model.generation_config.pad_token_id or 0)
    # Transformers 5.17 fills unspecified fields from model.generation_config.
    # Isolate the reference policy from saved sampling/logits-processor settings;
    # restore the caller's config even if generation fails. This benchmark is
    # single-threaded and owns the model for the duration of the comparison.
    saved_config = target.model.generation_config
    try:
        target.model.generation_config = config
        result = target.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
            generation_config=config)
    finally:
        target.model.generation_config = saved_config
    reference = result[0, ids.shape[1]:].tolist()
    return {"cached_tokens": cached, "hf_cached_tokens": reference, "equal": cached == reference}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--records")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", choices=("auto", "float32", "bfloat16"), default="auto")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--examples", type=int, default=7)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--check-recompute-tokens", type=int, default=16)
    parser.add_argument("--block-lengths", type=int, nargs="+", default=list(range(1, 9)))
    args = parser.parse_args()
    if min(args.examples, args.max_new_tokens, args.warmup, args.repeats,
           args.check_recompute_tokens, *args.block_lengths) < 1:
        parser.error("All counts, warmup and block lengths must be positive")
    output = Path(args.output)
    partial = output.with_suffix(".partial.json")
    if output.exists() or partial.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = (torch.bfloat16 if device.type == "cuda" else torch.float32) if args.dtype == "auto" else getattr(torch, args.dtype)
    torch.manual_seed(17)
    if args.tiny:
        if args.model or args.records:
            parser.error("--tiny cannot be combined with --model or --records")
        config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            max_position_embeddings=512, attention_dropout=0.0)
        config._attn_implementation = "sdpa"
        model = Qwen3ForCausalLM(config).to(device=device, dtype=dtype)
        rows = [{"id": "synthetic-tiny", "prompt_token_ids": list(range(1, 17))}]
        eos = set()
        model_hashes = {"config": hashlib.sha256(model.config.to_json_string().encode()).hexdigest()}
    else:
        if not args.model or not args.records:
            parser.error("Real model benchmarks require --model and --records")
        model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
            dtype=dtype,
            attn_implementation="sdpa").to(device)
        rows = [json.loads(line) for line in Path(args.records).read_text().splitlines()]
        rows = [r for r in rows if r["accepted"] and r["split"] == args.split][:args.examples]
        if not rows:
            raise ValueError("No accepted records in the requested split")
        configured_eos = model.generation_config.eos_token_id
        eos = set(configured_eos if isinstance(configured_eos, list) else
                  ([] if configured_eos is None else [configured_eos]))
        paths = sorted(set(Path(args.model).glob("*.safetensors")) |
                       {p for p in Path(args.model).glob("*.json")})
        model_hashes = {p.name: sha256(p) for p in paths}
    target = CachedTarget(model)
    rng = random.Random(17)
    generation, blocks, correctness, fidelity = [], [], [], []
    sources = [Path(__file__), Path(__file__).with_name("cached_target.py")]
    source_hashes = {p.name: sha256(p) for p in sources}
    def save_partial(status):
        # Preserve every completed trial and any failed correctness gate.
        # A partial run must never be mistaken for a successful cost report.
        data = {"status": status, "scope": "incomplete run; not a successful benchmark report",
            "config": vars(args), "source_sha256": source_hashes,
            "model_sha256": model_hashes, "correctness": correctness,
            "numerical_fidelity_full_vs_cached": fidelity,
            "generation": generation, "verification_blocks": blocks}
        temporary = partial.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2) + "\n")
        temporary.replace(partial)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for row in rows:
        ids = torch.tensor([row["prompt_token_ids"]], device=device, dtype=torch.long)
        correctness.append(dict(id=row["id"], **check_hf_cached(target, ids, args.check_recompute_tokens, eos)))
        if not correctness[-1]["equal"]:
            save_partial("failed_hf_cached_equality")
            raise RuntimeError(f"Handwritten/HF cached greedy mismatch; saved {partial}")
        fidelity.append(dict(id=row["id"], **check_recompute(target, ids, args.check_recompute_tokens, eos)))
        # Verification costs use target-greedy continuation tokens at a fixed
        # prefix. No acceptance distribution or draft overhead is represented.
        continuation = target.greedy(ids, max(args.block_lengths), eos_ids=())
        reference_tokens = None
        for trial in range(-args.warmup, args.repeats):
            result = generation_trial(target, ids, args.max_new_tokens, eos)
            if reference_tokens is None:
                reference_tokens = result["tokens"]
            elif result["tokens"] != reference_tokens:
                raise RuntimeError("Greedy tokens changed between timing trials")
            if trial >= 0:
                generation.append(dict(id=row["id"], prompt_tokens=ids.shape[1], trial=trial, **result))
            target.prefill(ids)
            lengths = list(args.block_lengths)
            rng.shuffle(lengths)
            for length in lengths:
                chunk = ids.new_tensor([continuation[:length]])
                def verify():
                    features = target.append(chunk)
                    # Full block vocabulary projection and greedy decisions,
                    # as needed by a future draft verifier.
                    return target.logits(features.last).argmax(-1).tolist()
                predictions, block_ms = timed(device, verify)
                _, crop_ms = timed(device, lambda: target.crop(ids.shape[1]))
                if trial >= 0:
                    blocks.append({"id": row["id"], "prompt_tokens": ids.shape[1], "trial": trial,
                        "new_tokens": length, "verify_ms": block_ms, "crop_ms": crop_ms,
                        "predictions": predictions[0]})
            if trial >= 0:
                save_partial("running")
        print(json.dumps({"id": row["id"], "completed_repeats": args.repeats}), flush=True)
    total_tokens = sum(r["output_tokens"] for r in generation)
    total_ms = sum(r["e2e_ms"] for r in generation)
    decode_tokens = sum(r["decode_steps"] for r in generation)
    decode_ms = sum(r["decode_remaining_ms"] for r in generation)
    report = {"scope": "cached target-only baseline and isolated verification cost; no speculative correctness or speedup claim",
        "correctness_scope": "CPU FP32 tests establish cache/rollback behavior; same-dtype HF cached greedy is the real-model token gate. "
            "Full-recompute comparisons are recorded separately without epsilon or sample exclusion. "
            "BF16 kernel-shape changes can alter argmax near ties; cached-block versus cached-sequential verification remains a separate gate.",
        "config": vars(args), "runtime": {"torch": torch.__version__, "transformers": transformers.__version__,
            "device": str(device), "dtype": str(next(model.parameters()).dtype),
            "attention_backend": "sdpa", "execution": "eager Transformers; no compile or CUDA graph",
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "hip": torch.version.hip, "cuda": torch.version.cuda,
            "threads": torch.get_num_threads()},
        "model_sha256": model_hashes, "source_sha256": source_hashes,
        "records_sha256": sha256(args.records) if args.records else None,
        "measurement": "perf_counter with device synchronization; e2e includes phase sync overhead; "
            "prefill_first_logit_ms includes LM head and argmax; decode measures remaining emitted tokens; "
            "verification includes append, all chunk logits, argmax and host copy; crop timed separately; "
            "verification lengths are new target inputs (anchor plus proposed tokens), not draft lengths; "
            "warmup excluded, block order shuffled at each repeat, fixed prompt prefixes; batch size 1; "
            "no intermediate target feature capture or draft costs included",
        "correctness": correctness, "numerical_fidelity_full_vs_cached": fidelity,
        "generation": generation, "verification_blocks": blocks,
        "summary": {"e2e_ms": distribution([r["e2e_ms"] for r in generation]),
            "prefill_first_logit_ms": distribution([r["prefill_first_logit_ms"] for r in generation]),
            "aggregate_e2e_tokens_per_second": total_tokens / (total_ms / 1000),
            "aggregate_decode_tokens_per_second": decode_tokens / (decode_ms / 1000) if decode_tokens else None,
            "verify_ms_by_new_tokens": {str(k): distribution([r["verify_ms"] for r in blocks if r["new_tokens"] == k])
                                       for k in sorted(set(args.block_lengths))},
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}}
    with output.open("x") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    partial.unlink()
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
