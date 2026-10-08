"""Load a real checkpoint and compare reference greedy decode token for token."""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from .checkpoint import load_checkpoint, sha256
from .config import DraftConfig
from .decode import speculative_greedy, target_greedy
from .model import DSparkDraft
from .target import FrozenTarget


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--examples", type=int, default=2)
    args = parser.parse_args()
    cp = Path(args.checkpoint)
    meta = json.loads((cp / "metadata.json").read_text())
    cfg = meta["identity"]["config"]
    for name, expected in meta["identity"]["target_fingerprint"].items():
        if sha256(Path(cfg["model"]) / name) != expected:
            raise ValueError(f"Target fingerprint mismatch: {name}")
    if sha256(cfg["records"]) != meta["identity"]["records_sha256"]:
        raise ValueError("Evaluation data hash mismatch")
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    spec = DraftConfig.from_dict(meta["draft_config"])
    target = FrozenTarget(model, spec)
    draft = DSparkDraft(model, spec).to("cuda")
    load_checkpoint(cp, draft)
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    records = [json.loads(x) for x in Path(cfg["records"]).read_text().splitlines()]
    records = [r for r in records if r["accepted"] and r["split"] == "validation"][:args.examples]
    results = []
    for row in records:
        ids = torch.tensor([row["prompt_token_ids"]], device="cuda")
        baseline = target_greedy(target, ids, args.max_new_tokens, eos)
        actual, rounds = speculative_greedy(target, draft, ids, args.max_new_tokens, eos, amp=True)
        result = {"id": row["id"], "equal": baseline == actual, "target_tokens": baseline,
            "speculative_tokens": actual, "rounds": rounds}
        results.append(result)
        print(json.dumps({"equal": result["equal"], "tokens": len(actual), "rounds": len(rounds)}), flush=True)
    report = {"all_equal": bool(results) and all(x["equal"] for x in results), "examples": results,
        "scope": "greedy correctness only; full-prefix recomputation, no KV cache, no speed benchmark"}
    (cp.parent / "greedy-evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["all_equal"]:
        raise RuntimeError("Greedy mismatch; inspect saved token traces")


if __name__ == "__main__":
    main()
