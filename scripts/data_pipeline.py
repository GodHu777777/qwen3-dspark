"""CPU-only identity, selection and split exports shared by the data CLIs."""
import hashlib
import json
import os
import random
import unicodedata
from collections import Counter
from pathlib import Path

SPLITS = ("train", "validation", "test")
TEST_POLICY = {"purpose": "final test", "checkpoint_selection_allowed": False,
               "policy_selection_allowed": False,
               "unlock_requires": "freeze checkpoint and decoding policy before final evaluation"}


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def prompt_id(messages):
    if not messages or messages[-1]["role"] != "user":
        raise ValueError("Expected first-user prompt")
    normalized = " ".join(unicodedata.normalize("NFKC", messages[-1]["content"]).split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def exclusions(paths):
    ids, provenance = set(), []
    for path in paths:
        rows = read_jsonl(path)
        for row in rows:
            messages = row["messages"]
            # Inputs may be prompt files or generated records; exclude the first user.
            end = next((i for i, m in enumerate(messages) if m["role"] == "user"), None)
            if end is None:
                raise ValueError("Exclusion record lacks user prompt")
            key = prompt_id(messages[:end + 1])
            if "id" in row and row["id"] != key:
                raise ValueError("Exclusion prompt ID mismatch")
            ids.add(key)
        provenance.append({"sha256": digest(path), "records": len(rows)})
    return ids, provenance


def split_counts(cfg):
    total, dev, test = cfg["samples"], cfg["validation_samples"], cfg.get("test_samples", 0)
    if any(type(x) is not int or x < 0 for x in (total, dev, test)) or total <= dev + test:
        raise ValueError("Invalid train/validation/test population")
    return {"train": total - dev - test, "validation": dev, "test": test}


def select_prompts(rows, tokenizer, cfg, excluded):
    split_counts(cfg)
    indices = list(range(len(rows)))
    random.Random(cfg["seed"]).shuffle(indices)
    selected, seen, skipped = [], set(), Counter()
    for idx in indices:
        row, messages = rows[idx], []
        for turn in row["conversations"]:
            role = {"human": "user", "user": "user", "system": "system"}.get(turn["from"])
            value = turn["value"]
            if role is None or not isinstance(value, str) or not value.strip():
                break
            messages.append({"role": role, "content": value})
            if role == "user":
                break
        if not messages or messages[-1]["role"] != "user":
            skipped["no_initial_user"] += 1
            continue
        key = prompt_id(messages)
        if key in excluded:
            skipped["excluded_prompt"] += 1
            continue
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
            "prompt_token_ids": list(tokens), "source": row.get("source"), "source_row": idx,
            "dataset": cfg["dataset"], "dataset_revision": cfg["dataset_revision"],
            "source_shard": cfg["shard"]})
        if len(selected) == cfg["samples"]:
            break
    if len(selected) != cfg["samples"]:
        raise ValueError(f"Only {len(selected)} eligible prompts, need {cfg['samples']}")
    for i, item in enumerate(selected):
        item["split"] = ("validation" if i < cfg["validation_samples"] else
            "test" if i < cfg["validation_samples"] + cfg.get("test_samples", 0) else "train")
    return selected, dict(skipped)


def validate_prompts(prompts, cfg, excluded=()):
    if len(prompts) != cfg["samples"]:
        raise ValueError("Input population mismatch")
    seen = set()
    for item in prompts:
        key = prompt_id(item["messages"])
        if key != item["id"] or key in seen or key in excluded:
            raise ValueError("Prompt identity collision or exclusion leak")
        seen.add(key)
        if item["split"] not in SPLITS:
            raise ValueError("Unknown split")
        if "prompt_token_ids" in item and len(item["prompt_token_ids"]) != item["prompt_tokens"]:
            raise ValueError("Prompt token count mismatch")
        if item["prompt_tokens"] > cfg["max_prompt_tokens"]:
            raise ValueError("Prompt length exceeds budget")
    actual = Counter(x["split"] for x in prompts)
    if any(actual[k] != v for k, v in split_counts(cfg).items()):
        raise ValueError("Split population mismatch")


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def completed_batch(path, expected_ids):
    """A crash between payload and checksum publication restarts the whole batch."""
    path = Path(path)
    checksum = path.with_suffix(".sha256")
    if not path.exists() or not checksum.exists():
        return None
    if checksum.read_text().strip() != digest(path):
        raise ValueError("Completed batch checksum mismatch")
    records = json.loads(path.read_text())
    if [x["id"] for x in records] != expected_ids:
        raise ValueError("Batch identity mismatch")
    return records


def write_jsonl(path, rows, compact=False):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as f:
        for row in rows:
            if compact:
                row = {"messages": row["messages"], "id": row["id"]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def export_records(out, records, cfg):
    """Keep final-test completions OUT of the training records path entirely."""
    out = Path(out)
    files = []
    for split in {x["split"] for x in records}:
        if split not in SPLITS:
            raise ValueError("Unknown export split")
    has_test = cfg.get("test_samples", 0) > 0
    development = [x for x in records if x["split"] != "test"]
    for name, rows, compact in [
        ("records.jsonl", development, False),
        ("rejected.jsonl", [x for x in development if not x["accepted"]], False),
        ("train.jsonl", [x for x in development if x["accepted"] and x["split"] == "train"], True),
        ("validation.jsonl", [x for x in development if x["accepted"] and x["split"] == "validation"], True)]:
        write_jsonl(out / name, rows, compact)
        files.append(name)
    if has_test:
        locked = out / "final-test"
        locked.mkdir(exist_ok=True, mode=0o700)
        locked.chmod(0o700)
        test = [x for x in records if x["split"] == "test"]
        for name, rows, compact in [("records.jsonl", test, False),
                ("test.jsonl", [x for x in test if x["accepted"]], True),
                ("rejected.jsonl", [x for x in test if not x["accepted"]], False)]:
            write_jsonl(locked / name, rows, compact)
            (locked / name).chmod(0o600)
            files.append("final-test/" + name)
        atomic_json(locked / "lock.json", TEST_POLICY)
        (locked / "lock.json").chmod(0o600)
        files.append("final-test/lock.json")
    return {name: digest(out / name) for name in files}
