"""Create an immutable, prompt-disjoint pilot from a pinned local Parquet shard."""
import argparse
import hashlib
import json
import random
import unicodedata
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--parquet", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], local_files_only=True)
    rows = pq.read_table(args.parquet).to_pylist()
    indices = list(range(len(rows)))
    random.Random(cfg["seed"]).shuffle(indices)
    selected, seen, skipped = [], set(), Counter()
    for idx in indices:
        row = rows[idx]
        messages = []
        for turn in row["conversations"]:
            role = {"human": "user", "user": "user", "system": "system"}.get(turn["from"])
            if role is None:
                break
            value = turn["value"]
            if not isinstance(value, str) or not value.strip():
                break
            messages.append({"role": role, "content": value})
            if role == "user":
                break
        if not messages or messages[-1]["role"] != "user":
            skipped["no_initial_user"] += 1
            continue
        # Split by normalized first-user prompt, even if system messages differ.
        normalized = " ".join(unicodedata.normalize("NFKC", messages[-1]["content"]).split())
        key = hashlib.sha256(normalized.encode()).hexdigest()
        if key in seen:
            skipped["duplicate_prompt"] += 1
            continue
        tokens = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
            add_generation_prompt=True, enable_thinking=cfg["enable_thinking"])
        if len(tokens) > cfg["max_prompt_tokens"]:
            skipped["long_prompt"] += 1
            continue
        seen.add(key)
        selected.append({"id": key, "messages": messages, "prompt_tokens": len(tokens),
            "source": row.get("source"), "source_row": idx,
            "dataset": cfg["dataset"], "dataset_revision": cfg["dataset_revision"],
            "source_shard": cfg["shard"]})
        if len(selected) == cfg["samples"]:
            break
    if len(selected) != cfg["samples"]:
        raise RuntimeError(f"Only {len(selected)} eligible prompts")
    for i, item in enumerate(selected):
        item["split"] = "validation" if i < cfg["validation_samples"] else "train"
    (out / "prompts.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in selected))
    manifest = {"config": cfg, "parquet_sha256": digest(args.parquet),
        "shard_rows": len(rows), "selection": "seeded shuffled row order, unique first-user prompts only",
        "scope": "one-shard pilot; original assistant turns and later conversation turns discarded",
        "skipped": dict(skipped), "selected_sources": dict(Counter(x["source"] for x in selected)),
        "splits": dict(Counter(x["split"] for x in selected)),
        "prompts_sha256": digest(out / "prompts.jsonl")}
    (out / "selection.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
