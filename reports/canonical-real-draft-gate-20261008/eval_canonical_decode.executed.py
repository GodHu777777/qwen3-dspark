"""Experimental real-draft canonical-target gate, not a speed benchmark.

Reuses every example of a supplied prior gate, including its known failures.
Token-bearing result files are private evidence. This optional fixed-query
control does not replace the variable-budget DSpark execution path.
"""
import argparse
import json
from pathlib import Path
import time

import torch
import transformers
from transformers import AutoModelForCausalLM

from .cached_decode import cached_speculative_greedy
from .cached_target import CachedTarget
from .canonical_target import CanonicalTarget
from .checkpoint import load_checkpoint, sha256
from .config import DraftConfig
from .model import DSparkDraft


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-gate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--capacity", type=int, default=256)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if min(args.max_new_tokens, args.capacity) < 1:
        parser.error("Token budget and capacity must be positive")
    prior = json.loads(Path(args.prior_gate).read_text())
    checkpoint = Path(prior["identity"]["args"]["checkpoint"])
    meta = json.loads((checkpoint / "metadata.json").read_text())
    cfg, spec = meta["identity"]["config"], DraftConfig.from_dict(meta["draft_config"])
    width = spec.block_size + 1
    records = {row["id"]: row for row in map(json.loads, Path(cfg["records"]).read_text().splitlines())}
    examples = prior["examples"]
    if not examples:
        raise ValueError("Prior gate has no examples")
    for example in examples:
        row = records[example["id"]]
        if len(row["prompt_token_ids"]) + args.max_new_tokens + width > args.capacity:
            raise ValueError("Capacity must include the full padded-query headroom for every example")
    for name, digest in meta["identity"]["target_fingerprint"].items():
        if sha256(Path(cfg["model"]) / name) != digest:
            raise ValueError(f"Target fingerprint mismatch: {name}")
    if sha256(cfg["records"]) != meta["identity"]["records_sha256"]:
        raise ValueError("Records fingerprint mismatch")
    if args.dry_run:
        print(json.dumps({"dry_run": True, "examples": len(examples), "width": width,
            "capacity": args.capacity, "original_failures": sum(not e["cached_equal"] for e in examples)}))
        return
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    free, total = torch.cuda.mem_get_info()
    if free < 6 * 1024**3:
        raise RuntimeError("Require at least 6 GiB free for the bounded real-draft gate")
    torch.cuda.set_per_process_memory_fraction(4 * 1024**3 / total)
    started = time.monotonic()
    identity = {"args": vars(args), "prior_gate_sha256": sha256(args.prior_gate),
        "checkpoint_sha256": sha256(checkpoint / "draft.safetensors"),
        "checkpoint_metadata_sha256": sha256(checkpoint / "metadata.json"),
        "source_sha256": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
        "torch": torch.__version__, "transformers": transformers.__version__, "hip": torch.version.hip,
        "target_dtype": "bfloat16", "draft_dtype": "float32 parameters / bfloat16 AMP",
        "backend": "sdpa", "query_width": width, "capacity": args.capacity,
        "prefill_policy": "shared full prompt forward and full-row LM head, then select last row",
        "scope": "experimental fixed-query real-draft correctness gate; no throughput measurement"}
    (out / "run.json").write_text(json.dumps(identity, indent=2) + "\n")
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    draft = DSparkDraft(model, spec).to("cuda").eval()
    load_checkpoint(checkpoint, draft)
    stock = CachedTarget(model)
    canonical = CanonicalTarget(model, spec.layer_ids, capacity=args.capacity, query_width=width)
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    results = []
    for number, original in enumerate(examples):
        row = records[original["id"]]
        ids = torch.tensor([row["prompt_token_ids"]], device="cuda", dtype=torch.long)
        baseline = canonical.greedy(ids, args.max_new_tokens, eos)
        actual, rounds = cached_speculative_greedy(canonical, draft, ids, args.max_new_tokens,
            eos, amp=True, trace_tokens=True)
        canonical.validate_cache()
        expected_cache = ids.shape[1] + len(actual) - 1
        if canonical.length != expected_cache:
            raise RuntimeError("Final canonical target cache does not exclude the last emitted anchor")
        conventional = stock.greedy(ids, args.max_new_tokens, eos)
        result = {"example_index": number, "id": row["id"], "split": row["split"],
            "original_cached_equal": original["cached_equal"], "prompt_tokens": ids.shape[1],
            "canonical_equal": baseline == actual, "stock_equal": conventional == actual,
            "canonical_stock_equal": baseline == conventional,
            "canonical_tokens": baseline, "speculative_tokens": actual, "stock_tokens": conventional,
            "rounds": rounds, "final_cache_length": canonical.length}
        if baseline != actual:
            result["first_canonical_mismatch_index"] = next(
                (i for i, (a, b) in enumerate(zip(baseline, actual)) if a != b), min(len(baseline), len(actual)))
        results.append(result)
        report = {"identity": identity, "examples": results,
            "all_canonical_equal": all(r["canonical_equal"] for r in results),
            "all_stock_equal": all(r["stock_equal"] for r in results),
            "elapsed_seconds_not_benchmark": time.monotonic() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
        temporary = out / "result.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(out / "result.json")
        print(json.dumps({"example_index": number, "canonical_equal": result["canonical_equal"],
            "stock_equal": result["stock_equal"], "output_tokens": len(actual),
            "rounds": len(rounds), "committed_accepted": sum(r["committed_accepted"] for r in rounds)}), flush=True)
    if not report["all_canonical_equal"]:
        raise RuntimeError("Real draft failed canonical target equality; private traces saved")


if __name__ == "__main__":
    main()
