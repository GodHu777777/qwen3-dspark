"""Read-only development-data checks for standalone resource/evaluation tools."""
import json
from pathlib import Path

import torch
import transformers

from .checkpoint import sha256
from .data import load_records


def development_rows(path, split, max_sequence_tokens):
    if split not in ("train", "validation"):
        raise ValueError("Only train/validation are allowed; final test is locked")
    seen = set()
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["split"] not in ("train", "validation"):
                raise ValueError("Final test or unknown split in development records")
            if row["id"] in seen:
                raise ValueError("Duplicate development identity or train/validation leakage")
            seen.add(row["id"])
            if row["accepted"]:
                if not row["prompt_token_ids"] or not row["output_token_ids"]:
                    raise ValueError("Accepted record has an empty prompt or completion")
                if any(type(token) is not int or token < 0 for token in
                       row["prompt_token_ids"] + row["output_token_ids"]):
                    raise ValueError("Invalid token IDs in accepted record")
    # Retain the exact accepted/EOS/template/sequence-budget semantics used by
    # training, with stricter final-test rejection at this tool boundary.
    return load_records(path, split, max_sequence_tokens)


def verify_identity(cfg, expected=None):
    manifest_path = Path(cfg["generation_manifest"])
    summary_path = manifest_path.with_name("summary.json")
    generation = json.loads(manifest_path.read_text())
    summary = json.loads(summary_path.read_text())
    records_hash = sha256(cfg["records"])
    if records_hash != summary["output_sha256"]["records.jsonl"]:
        raise ValueError("Development records differ from generation evidence")
    fingerprint = generation["model_files_sha256"]
    for name, digest in fingerprint.items():
        if sha256(Path(cfg["model"]) / name) != digest:
            raise ValueError(f"Target/tokenizer fingerprint mismatch: {name}")
    if expected is not None:
        if records_hash != expected["records_sha256"] or fingerprint != expected["target_fingerprint"]:
            raise ValueError("Data/target identity differs from checkpoint training identity")
    return {"config": cfg, "records_sha256": records_hash, "target_fingerprint": fingerprint,
        "generation_manifest_sha256": sha256(manifest_path),
        "generation_summary_sha256": sha256(summary_path),
        "source_sha256": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "torch": torch.__version__, "transformers": transformers.__version__, "hip": torch.version.hip}
