"""Immutable prompt selection from one pinned Parquet shard (CPU only)."""
import argparse
import json
import os
import platform
import tempfile
from collections import Counter
from pathlib import Path

try:
    from .data_pipeline import digest, exclusions, select_prompts, validate_prompts, write_jsonl, atomic_json, read_jsonl
except ImportError:
    from data_pipeline import digest, exclusions, select_prompts, validate_prompts, write_jsonl, atomic_json, read_jsonl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--parquet", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--exclude-prompts", action="append", default=[])
    p.add_argument("--resume", action="store_true", help="Verify and reuse a completed immutable selection")
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    excluded, excluded_files = exclusions(args.exclude_prompts)
    if cfg.get("require_exclusions") and not excluded:
        raise ValueError("Expansion requires nonempty prior-prompt exclusions")
    parquet_hash = digest(args.parquet)
    if cfg.get("parquet_sha256") and cfg["parquet_sha256"] != parquet_hash:
        raise ValueError("Pinned Parquet hash mismatch")
    import pyarrow
    import pyarrow.parquet as pq
    import transformers
    from transformers import AutoTokenizer
    model = Path(cfg["model"])
    tokenizer_hashes = {x.name: digest(x) for x in sorted(model.iterdir())
        if x.suffix in {".json", ".txt", ".jinja"}}
    identity = {"config": cfg, "config_sha256": digest(args.config), "parquet_sha256": parquet_hash,
        "excluded_files": excluded_files, "excluded_prompt_count": len(excluded),
        "script_sha256": digest(__file__), "pipeline_sha256": digest(Path(__file__).with_name("data_pipeline.py")),
        "tokenizer_files_sha256": tokenizer_hashes, "python": platform.python_version(),
        "pyarrow": pyarrow.__version__, "transformers": transformers.__version__}
    out = Path(args.output)
    if out.exists():
        if not args.resume:
            raise ValueError("Selection already exists; use --resume to verify, or a new directory")
        previous = json.loads((out / "selection.json").read_text())
        if previous["identity"] != identity:
            raise ValueError("Selection resume identity mismatch")
        for name, sha in previous["output_sha256"].items():
            if digest(out / name) != sha:
                raise ValueError("Immutable selection changed")
        validate_prompts(read_jsonl(out / "prompts.jsonl"), cfg, excluded)
        print(json.dumps({"resumed": True, "splits": previous["splits"], "prompts_sha256": previous["prompts_sha256"]}))
        return
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    rows = pq.read_table(args.parquet).to_pylist()
    selected, skipped = select_prompts(rows, tokenizer, cfg, excluded)
    validate_prompts(selected, cfg, excluded)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Publish only a complete selection; failed preparation leaves no output directory.
    with tempfile.TemporaryDirectory(prefix=out.name + ".preparing-", dir=out.parent) as temp:
        staging = Path(temp) / "input"
        staging.mkdir(mode=0o700)
        write_jsonl(staging / "prompts.jsonl", selected)
        atomic_json(staging / "excluded-prompt-ids.json", sorted(excluded))
        manifest = {"config": cfg, "identity": identity, "parquet_sha256": parquet_hash,
            "shard_rows": len(rows), "selection": "seeded shuffled row order, normalized first-user identity",
            "scope": "one shard; original answers/later turns discarded; validation is dev; test locked",
            "skipped": skipped, "selected_sources": dict(Counter(x["source"] for x in selected)),
            "splits": dict(Counter(x["split"] for x in selected)),
            "prompts_sha256": digest(staging / "prompts.jsonl"),
            "output_sha256": {x: digest(staging / x) for x in ["prompts.jsonl", "excluded-prompt-ids.json"]}}
        atomic_json(staging / "selection.json", manifest)
        os.rename(staging, out)
    print(json.dumps({"splits": manifest["splits"], "skipped": skipped,
        "excluded_prompt_count": len(excluded), "prompts_sha256": manifest["prompts_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
