"""Audit exported data against prompts and exact generation records (stdlib only)."""
import argparse
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    args = p.parse_args()
    run = Path(args.run)
    prompts = read(run / "input/prompts.jsonl")
    records = read(run / "generated/records.jsonl")
    train = read(run / "generated/train.jsonl")
    val = read(run / "generated/validation.jsonl")
    rejected = read(run / "generated/rejected.jsonl")
    assert len({x["id"] for x in prompts}) == len(prompts)
    assert [x["id"] for x in prompts] == [x["id"] for x in records]
    assert not ({x["id"] for x in train} & {x["id"] for x in val})
    assert len(train) + len(val) + len(rejected) == len(records)
    by_id = {x["id"]: x for x in records}
    for prompt, row in zip(prompts, records):
        normalized = " ".join(unicodedata.normalize("NFKC", prompt["messages"][-1]["content"]).split())
        assert hashlib.sha256(normalized.encode()).hexdigest() == prompt["id"]
        assert row["messages"][:-1] == prompt["messages"]
        assert all(m["role"] in {"system", "user"} for m in prompt["messages"])
        assert row["messages"][-1]["role"] == "assistant"
        assert row["prompt_tokens"] == len(row["prompt_token_ids"])
        assert row["generated_tokens"] == len(row["output_token_ids"])
        assert row["accepted"] == (not row["rejection_reasons"])
    for split, rows in [("train", train), ("validation", val)]:
        for row in rows:
            record = by_id[row["id"]]
            assert record["accepted"] and record["split"] == split
            assert row["messages"] == record["messages"]
            assert record["finish_reason"] == "eos"
            assert record["template_prefix_matches_generation"]
            assert record["training_tokens"] <= 4096
    summary = json.loads((run / "generated/summary.json").read_text())
    for name, expected in summary["output_sha256"].items():
        assert hashlib.sha256((run / "generated" / name).read_bytes()).hexdigest() == expected
    report = {"passed": True, "records": len(records), "train": len(train),
        "validation": len(val), "rejected": len(rejected),
        "checks": ["prompt identity and disjoint splits", "no source assistant turns in prompts",
            "messages match records", "token lengths", "EOS and template alignment",
            "sequence budget", "output SHA256"],
        "accepted_sources": dict(Counter(x["source"] for x in records if x["accepted"]))}
    (run / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
