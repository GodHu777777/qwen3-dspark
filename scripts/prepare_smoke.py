"""Mechanically select a small three-way smoke; never inspect/score test content."""
import argparse
import copy
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

try:
    from .data_pipeline import atomic_json, digest, read_jsonl, validate_prompts, write_jsonl
except ImportError:
    from data_pipeline import atomic_json, digest, read_jsonl, validate_prompts, write_jsonl


def prepare_subset(parent_dir, out, per_split=2):
    parent_dir, out = Path(parent_dir), Path(out)
    if per_split < 1:
        raise ValueError("Smoke requires each split")
    selection = json.loads((parent_dir / "selection.json").read_text())
    for name, expected in selection["output_sha256"].items():
        if digest(parent_dir / name) != expected:
            raise ValueError("Parent selection changed")
    parent_rows = read_jsonl(parent_dir / "prompts.jsonl")
    excluded = json.loads((parent_dir / "excluded-prompt-ids.json").read_text())
    validate_prompts(parent_rows, selection["config"], excluded)
    rows = []
    # Selection is by stored split/order only. No length, quality or answer inspection.
    for split in ("validation", "test", "train"):
        candidates = [x for x in parent_rows if x["split"] == split]
        if len(candidates) < per_split:
            raise ValueError("Insufficient smoke split population")
        rows.extend(candidates[:per_split])
    cfg = copy.deepcopy(selection["config"])
    cfg.update(samples=3 * per_split, validation_samples=per_split,
        test_samples=per_split, batch_size=3 * per_split)
    validate_prompts(rows, cfg, excluded)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise ValueError("Smoke directory exists: preserve it and choose a new run")
    with tempfile.TemporaryDirectory(prefix=out.name + ".preparing-", dir=out.parent) as tmp:
        staging = Path(tmp) / "run"
        inp = staging / "input"
        inp.mkdir(parents=True, mode=0o700)
        staging.chmod(0o700)
        atomic_json(staging / "smoke.local.json", cfg)
        write_jsonl(inp / "prompts.jsonl", rows)
        atomic_json(inp / "excluded-prompt-ids.json", excluded)
        identity = copy.deepcopy(selection["identity"])
        identity.update(config=cfg, config_sha256=digest(staging / "smoke.local.json"),
            script_sha256=digest(__file__), pipeline_sha256=digest(Path(__file__).with_name("data_pipeline.py")))
        manifest = {"config": cfg, "identity": identity,
            "parquet_sha256": selection["parquet_sha256"],
            "scope": "three-way smoke only; test content not used for selection or tuning",
            "parent_selection_sha256": digest(parent_dir / "selection.json"),
            "parent_prompts_sha256": digest(parent_dir / "prompts.jsonl"),
            "splits": dict(Counter(x["split"] for x in rows)),
            "prompts_sha256": digest(inp / "prompts.jsonl"),
            "output_sha256": {name: digest(inp / name) for name in ["prompts.jsonl", "excluded-prompt-ids.json"]}}
        atomic_json(inp / "selection.json", manifest)
        os.rename(staging, out)
    return {"splits": manifest["splits"], "parent_prompts_sha256": manifest["parent_prompts_sha256"],
        "prompts_sha256": manifest["prompts_sha256"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Completed parent selection directory")
    p.add_argument("--output", required=True, help="Fresh smoke run directory")
    p.add_argument("--per-split", type=int, default=2)
    args = p.parse_args()
    print(json.dumps(prepare_subset(args.input, args.output, args.per_split), indent=2))


if __name__ == "__main__":
    main()
