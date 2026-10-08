"""Real checkpoint correctness gate; saves private token traces, never times speed."""
import argparse
import json
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM

from .cached_decode import cached_speculative_greedy
from .cached_target import CachedTarget
from .checkpoint import load_checkpoint, sha256
from .config import DraftConfig
from .decode import target_greedy
from .model import DSparkDraft
from .target import FrozenTarget


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dev-examples", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()
    if args.dev_examples < 0 or args.max_new_tokens < 1:
        parser.error("Require nonnegative dev examples and positive token budget")
    cp = Path(args.checkpoint)
    meta = json.loads((cp / "metadata.json").read_text())
    cfg = meta["identity"]["config"]
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    for name, digest in meta["identity"]["target_fingerprint"].items():
        if sha256(Path(cfg["model"]) / name) != digest:
            raise ValueError("Target fingerprint mismatch")
    if sha256(cfg["records"]) != meta["identity"]["records_sha256"]:
        raise ValueError("Evaluation data identity mismatch")
    identity = {"args": vars(args), "checkpoint_weights_sha256": sha256(cp / "draft.safetensors"),
        "checkpoint_metadata_sha256": sha256(cp / "metadata.json"),
        "source_sha256": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "torch": torch.__version__, "transformers": transformers.__version__, "hip": torch.version.hip,
        "target_dtype": "bfloat16", "draft_compute": "bfloat16 AMP", "timing": "not measured"}
    (out / "run.json").write_text(json.dumps(identity, indent=2) + "\n")
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    spec = DraftConfig.from_dict(meta["draft_config"])
    draft = DSparkDraft(model, spec).to("cuda").eval()
    load_checkpoint(cp, draft)
    target, full = CachedTarget(model, spec.layer_ids), FrozenTarget(model, spec)
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    records = [json.loads(x) for x in Path(cfg["records"]).read_text().splitlines()]
    train = [r for r in records if r["id"] == meta.get("record_id")]
    rows = train + [r for r in records if r["accepted"] and r["split"] == "validation"][:args.dev_examples]
    results = []
    for row in rows:
        ids = torch.tensor([row["prompt_token_ids"]], device="cuda", dtype=torch.long)
        sequential = target.greedy(ids, args.max_new_tokens, eos)
        recompute = target_greedy(full, ids, args.max_new_tokens, eos)
        actual, rounds = cached_speculative_greedy(target, draft, ids, args.max_new_tokens,
            eos, amp=True, trace_tokens=True)
        result = {"id": row["id"], "split": row["split"], "cached_equal": sequential == actual,
            "recompute_equal": recompute == actual, "sequential_recompute_equal": sequential == recompute,
            "cached_target_tokens": sequential, "full_recompute_tokens": recompute,
            "speculative_tokens": actual, "rounds": rounds, "prompt_tokens": ids.shape[1]}
        if sequential != actual:
            index = next((i for i, pair in enumerate(zip(sequential, actual)) if pair[0] != pair[1]),
                min(len(sequential), len(actual)))
            result["first_mismatch_index"] = index
            # Reproduce the sequential target's exact same prefix/chunk schedule.
            features = target.prefill(ids)
            for value in sequential[:index]:
                features = target.append(ids.new_tensor([[value]]))
            logits = target.logits(features.last[:, -1]).float()
            values, indices = logits.topk(5, dim=-1)
            result["same_prefix_sequential"] = {"prefix_length": ids.shape[1] + index,
                "top_tokens": indices[0].tolist(), "top_values": values[0].tolist()}
            emitted = 1
            for round_index, r in enumerate(rounds):
                if emitted <= index < emitted + r["committed"]:
                    offset = index - emitted
                    result["same_prefix_block"] = {"round": round_index, "position_in_block": offset,
                        "prefix_length": r["cache_before"] + offset + 1,
                        "top_tokens": r["verified_top_tokens"][offset],
                        "top_values": r["verified_top_values"][offset]}
                    break
                emitted += r["committed"]
        results.append(result)
        report = {"identity": identity, "all_cached_equal": all(r["cached_equal"] for r in results),
            "examples": results, "scope": "BF16 cached-block correctness gate; no speed benchmark"}
        (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: result[k] for k in ("split", "cached_equal", "recompute_equal",
            "sequential_recompute_equal")}), flush=True)
    if not results or not report["all_cached_equal"]:
        raise RuntimeError("Cached speculative mismatch; saved same-prefix traces before failing")


if __name__ == "__main__":
    main()
