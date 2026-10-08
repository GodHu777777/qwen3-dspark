"""Audit normalized identities, provenance, exact records and locked final test."""
import argparse
import json
from collections import Counter
from pathlib import Path

try:
    from .data_pipeline import digest, read_jsonl, validate_prompts, TEST_POLICY
except ImportError:
    from data_pipeline import digest, read_jsonl, validate_prompts, TEST_POLICY


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_run(run):
    run = Path(run)
    selection = json.loads((run / "input/selection.json").read_text())
    manifest = json.loads((run / "generated/manifest.json").read_text())
    cfg = manifest["config"]
    prompts = read_jsonl(run / "input/prompts.jsonl")
    require(selection["config"] == cfg, "Selection and generation configuration mismatch")
    if cfg.get("parquet_sha256"):
        require(selection["parquet_sha256"] == cfg["parquet_sha256"], "Pinned source shard identity mismatch")
    if "identity" in selection and "config_sha256" in manifest:
        require(selection["identity"]["config_sha256"] == manifest["config_sha256"], "Configuration bytes changed")
    require(digest(run / "input/prompts.jsonl") == selection["prompts_sha256"] == manifest["prompts_sha256"],
        "Prompt bytes changed")
    if "selection_sha256" in manifest:
        require(digest(run / "input/selection.json") == manifest["selection_sha256"], "Selection identity changed")
    excluded_path = run / "input/excluded-prompt-ids.json"
    excluded = json.loads(excluded_path.read_text()) if excluded_path.exists() else []
    for name, expected in selection.get("output_sha256", {}).items():
        require(digest(run / "input" / name) == expected, "Selection artifact hash mismatch")
    require(not cfg.get("require_exclusions") or bool(excluded), "Missing exclusions")
    validate_prompts(prompts, cfg, excluded)
    development = read_jsonl(run / "generated/records.jsonl")
    split_files = {s: read_jsonl(run / "generated" / (s + ".jsonl")) for s in ["train", "validation"]}
    rejected = read_jsonl(run / "generated/rejected.jsonl")
    final = []
    if cfg.get("test_samples", 0):
        locked = run / "generated/final-test"
        final = read_jsonl(locked / "records.jsonl")
        split_files["test"] = read_jsonl(locked / "test.jsonl")
        rejected += read_jsonl(locked / "rejected.jsonl")
        require(json.loads((locked / "lock.json").read_text()) == TEST_POLICY, "Final test policy changed")
        require(all(x["split"] == "test" for x in final), "Non-test record in final test")
    require(all(x["split"] != "test" for x in development), "Final test leaked into training records path")
    by_id = {x["id"]: x for x in development + final}
    require(len(by_id) == len(development) + len(final) == len(prompts), "Duplicate/missing output identity")
    records = [by_id[x["id"]] for x in prompts]
    require([x["id"] for x in development] == [x["id"] for x in prompts if x["split"] != "test"],
        "Development record order changed")
    require([x["id"] for x in final] == [x["id"] for x in prompts if x["split"] == "test"],
        "Final test record order changed")
    summary = json.loads((run / "generated/summary.json").read_text())
    for prompt, row in zip(prompts, records):
        require(row["split"] == prompt["split"], "Split assignment changed")
        require(row["messages"][:-1] == prompt["messages"], "Prompt messages changed")
        require(all(m["role"] in {"system", "user"} for m in prompt["messages"]), "Source assistant in prompt")
        require(row["messages"][-1]["role"] == "assistant", "Missing assistant")
        require(row["prompt_tokens"] == len(row["prompt_token_ids"]) == prompt["prompt_tokens"], "Prompt token count")
        if "prompt_token_ids" in prompt:
            require(row["prompt_token_ids"] == prompt["prompt_token_ids"], "Exact prompt tokens changed")
        require(row["generated_tokens"] == len(row["output_token_ids"]), "Output token count")
        require(row["generated_tokens"] <= cfg["max_new_tokens"], "Output token budget")
        require(row["accepted"] == (not row["rejection_reasons"]), "Acceptance reasons mismatch")
        for key in ["dataset", "dataset_revision", "source_shard", "source_row", "source"]:
            require(row[key] == prompt[key], "Record source provenance changed")
    for split, rows in split_files.items():
        expected = [x for x in records if x["accepted"] and x["split"] == split]
        require([x["id"] for x in rows] == [x["id"] for x in expected], "Accepted split population/order mismatch")
        for row, record in zip(rows, expected):
            require(row["messages"] == record["messages"], "Exported messages mismatch")
            require(record["finish_reason"] == "eos" and record["template_prefix_matches_generation"], "Incomplete or misaligned response")
            require(record["training_tokens"] <= cfg["max_sequence_tokens"], "Sequence token budget")
            require(len(record["prompt_token_ids"]) + len(record["output_token_ids"]) <= cfg["max_sequence_tokens"],
                "Exact sequence token budget")
            if "eos_token_ids" in summary:
                require(record["output_token_ids"][-1] in summary["eos_token_ids"], "Missing EOS token")
    require({x["id"] for x in rejected} == {x["id"] for x in records if not x["accepted"]}, "Rejected population mismatch")
    require(len(rejected) == sum(not x["accepted"] for x in records), "Duplicate rejection")
    for row in rejected:
        require(row == by_id[row["id"]], "Rejected record changed")
    expected_files = {"train.jsonl", "validation.jsonl", "records.jsonl"}
    if cfg.get("test_samples", 0):
        expected_files |= {"rejected.jsonl", "final-test/records.jsonl", "final-test/test.jsonl",
            "final-test/rejected.jsonl", "final-test/lock.json"}
        require(summary.get("final_test_locked") is True, "Missing final test lock in summary")
    require(expected_files <= set(summary["output_sha256"]), "Missing export hash")
    for name, sha in summary["output_sha256"].items():
        require(digest(run / "generated" / name) == sha, "Output SHA256 mismatch")
    require(summary["total"] == len(records) and summary["accepted"] == sum(x["accepted"] for x in records), "Summary population")
    require(summary["accepted_splits"] == dict(Counter(x["split"] for x in records if x["accepted"])), "Summary split counts")
    return {"passed": True, "records": len(records), "train": len(split_files["train"]),
        "validation": len(split_files["validation"]), "test": len(split_files.get("test", [])),
        "rejected": len(rejected), "excluded_prompt_count": len(excluded),
        "final_test_locked": bool(cfg.get("test_samples", 0)),
        "checks": ["normalized prompt/exclusion identity and disjoint three-way splits", "source provenance",
            "no final test in training records", "exact tokens, EOS, sequence budget", "export populations and SHA256"],
        "accepted_sources": dict(Counter(x["source"] for x in records if x["accepted"]))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    args = p.parse_args()
    report = audit_run(args.run)
    (Path(args.run) / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
