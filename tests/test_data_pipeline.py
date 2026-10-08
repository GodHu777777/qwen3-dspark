"""Data isolation/identity tests; synthetic fixtures contain no dataset samples."""
import copy
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from scripts.audit import audit_run
from scripts.prepare_smoke import prepare_subset
from scripts.data_pipeline import (atomic_json, completed_batch, digest, exclusions,
    export_records, prompt_id, read_jsonl, select_prompts, split_counts, validate_prompts, write_jsonl)


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return list(range(len(messages[-1]["content"])))


def config():
    return {"samples": 6, "validation_samples": 1, "test_samples": 2, "seed": 4,
        "enable_thinking": False, "max_prompt_tokens": 64, "max_sequence_tokens": 128,
        "max_new_tokens": 16, "dataset": "synthetic", "dataset_revision": "pinned-test-revision",
        "shard": "synthetic.parquet", "require_exclusions": True}


def source_rows():
    return [{"source": "synthetic", "conversations": [{"from": "system", "value": "fixture"},
        {"from": "human", "value": f"item-{i}"}, {"from": "gpt", "value": "discarded-fixture"}]}
        for i in range(6)]


def fixture(run):
    cfg = config()
    out = run / "generated"
    inp = run / "input"
    out.mkdir(); inp.mkdir()
    excluded = [prompt_id([{"role": "user", "content": "previous-fixture"}])]
    prompts, _ = select_prompts(source_rows(), Tokenizer(), cfg, excluded)
    write_jsonl(inp / "prompts.jsonl", prompts)
    atomic_json(inp / "excluded-prompt-ids.json", excluded)
    selection = {"config": cfg, "prompts_sha256": digest(inp / "prompts.jsonl"),
        "identity": {"config_sha256": "fixture-config-hash"}, "parquet_sha256": "fixture-source-hash",
        "output_sha256": {name: digest(inp / name) for name in ["prompts.jsonl", "excluded-prompt-ids.json"]}}
    atomic_json(inp / "selection.json", selection)
    atomic_json(out / "manifest.json", {"config": cfg, "prompts_sha256": selection["prompts_sha256"],
        "selection_sha256": digest(inp / "selection.json")})
    rejected_id = next(x["id"] for x in prompts if x["split"] == "train")
    records = [{**x, "messages": x["messages"] + [{"role": "assistant", "content": "fixture-answer"}],
        "output_token_ids": [42, 99], "generated_tokens": 2,
        "training_tokens": x["prompt_tokens"] + 2, "finish_reason": "eos",
        "template_prefix_matches_generation": True, "accepted": x["id"] != rejected_id,
        "rejection_reasons": ["fixture-reject"] if x["id"] == rejected_id else []} for x in prompts]
    hashes = export_records(out, records, cfg)
    atomic_json(out / "summary.json", {"total": len(records), "accepted": 5,
        "accepted_splits": dict(Counter(x["split"] for x in records if x["accepted"])),
        "eos_token_ids": [99], "final_test_locked": True, "output_sha256": hashes})
    return cfg, prompts, records


def rehash_outputs(run):
    path = run / "generated/summary.json"
    summary = json.loads(path.read_text())
    summary["output_sha256"] = {name: digest(run / "generated" / name) for name in summary["output_sha256"]}
    atomic_json(path, summary)


class DataPipelineTests(unittest.TestCase):
    def test_normalized_exclusions_and_duplicate_differing_system(self):
        cfg = config()
        excluded_messages = [{"role": "user", "content": "FULL width"}]
        key = prompt_id(excluded_messages)
        rows = source_rows() + [{"source": "synthetic", "conversations": [
            {"from": "human", "value": "ＦＵＬＬ   width"}]}]
        rows.append({"source": "synthetic", "conversations": [
            {"from": "system", "value": "different-system"}, {"from": "human", "value": "item-0"}]})
        # Need all six eligible rows: exclusions and duplicate cannot hide behind early termination.
        cfg["samples"] = 7
        with self.assertRaisesRegex(ValueError, "Only 6 eligible"):
            select_prompts(rows, Tokenizer(), cfg, {key})
        cfg["samples"] = 6
        first, _ = select_prompts(rows, Tokenizer(), cfg, {key})
        second, _ = select_prompts(rows, Tokenizer(), cfg, {key})
        self.assertEqual(first, second)
        self.assertNotIn(key, {x["id"] for x in first})
        validate_prompts(first, cfg, {key})
        self.assertEqual(Counter(x["split"] for x in first), split_counts(cfg))
        self.assertTrue(all(x["messages"][-1]["role"] == "user" for x in first))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "previous.jsonl"
            write_jsonl(path, [{"id": key, "messages": excluded_messages}])
            ids, provenance = exclusions([path])
            self.assertEqual(ids, {key})
            self.assertEqual(provenance[0]["sha256"], digest(path))
            write_jsonl(path, [{"id": "wrong", "messages": excluded_messages}])
            with self.assertRaisesRegex(ValueError, "ID mismatch"):
                exclusions([path])

    def test_three_way_export_removes_test_from_training_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            _, _, records = fixture(run)
            report = audit_run(run)
            self.assertEqual((report["train"], report["validation"], report["test"]), (2, 1, 2))
            development = read_jsonl(run / "generated/records.jsonl")
            test = read_jsonl(run / "generated/final-test/records.jsonl")
            self.assertFalse({x["id"] for x in development} & {x["id"] for x in test})
            self.assertEqual(len(development) + len(test), len(records))
            self.assertEqual((run / "generated/final-test").stat().st_mode & 0o777, 0o700)
            self.assertEqual((run / "generated/final-test/records.jsonl").stat().st_mode & 0o777, 0o600)
            smoke = run / "smoke"
            result = prepare_subset(run / "input", smoke, per_split=1)
            self.assertEqual(result["splits"], {"validation": 1, "test": 1, "train": 1})
            subset = read_jsonl(smoke / "input/prompts.jsonl")
            self.assertEqual({x["id"] for x in subset},
                {next(x["id"] for x in read_jsonl(run / "input/prompts.jsonl") if x["split"] == s)
                 for s in ("train", "validation", "test")})
            with self.assertRaisesRegex(ValueError, "directory exists"):
                prepare_subset(run / "input", smoke, per_split=1)

    def test_audit_detects_test_leak_even_when_rehashed(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp); fixture(run)
            rows = read_jsonl(run / "generated/records.jsonl")
            rows += read_jsonl(run / "generated/final-test/records.jsonl")[:1]
            write_jsonl(run / "generated/records.jsonl", rows)
            rehash_outputs(run)
            with self.assertRaisesRegex(ValueError, "Final test leaked"):
                audit_run(run)

    def test_audit_detects_same_length_prompt_token_substitution(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp); fixture(run)
            rows = read_jsonl(run / "generated/records.jsonl")
            rows[0]["prompt_token_ids"][0] += 1
            write_jsonl(run / "generated/records.jsonl", rows); rehash_outputs(run)
            with self.assertRaisesRegex(ValueError, "Exact prompt tokens changed"):
                audit_run(run)

    def test_audit_detects_eos_and_export_identity_damage(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp); fixture(run)
            rows = read_jsonl(run / "generated/final-test/records.jsonl")
            rows[0]["output_token_ids"][-1] = 55
            write_jsonl(run / "generated/final-test/records.jsonl", rows); rehash_outputs(run)
            with self.assertRaisesRegex(ValueError, "Missing EOS"):
                audit_run(run)
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp); fixture(run)
            rows = read_jsonl(run / "generated/train.jsonl")
            rows.append(copy.deepcopy(rows[0]))
            write_jsonl(run / "generated/train.jsonl", rows); rehash_outputs(run)
            with self.assertRaisesRegex(ValueError, "split population"):
                audit_run(run)

    def test_audit_rejects_exclusion_leak_and_selection_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp); _, prompts, _ = fixture(run)
            path = run / "input/selection.json"
            selection = json.loads(path.read_text()); selection["config"]["seed"] += 1
            atomic_json(path, selection)
            with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                audit_run(run)
        cfg = config(); selected, _ = select_prompts(source_rows(), Tokenizer(), cfg, set())
        with self.assertRaisesRegex(ValueError, "exclusion leak"):
            validate_prompts(selected, cfg, {selected[0]["id"]})

    def test_batch_crash_resume_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "000000.json"
            self.assertIsNone(completed_batch(path, ["fixture"]))
            atomic_json(path, [{"id": "fixture"}])
            self.assertIsNone(completed_batch(path, ["fixture"]))
            path.with_suffix(".sha256").write_text(digest(path) + "\n")
            self.assertEqual(completed_batch(path, ["fixture"]), [{"id": "fixture"}])
            with self.assertRaisesRegex(ValueError, "Batch identity"):
                completed_batch(path, ["wrong"])
            path.write_text('[{"id":"corrupt"}]')
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                completed_batch(path, ["fixture"])

    def test_invalid_split_population_rejected(self):
        cfg = config(); cfg["test_samples"] = cfg["samples"]
        with self.assertRaisesRegex(ValueError, "population"):
            split_counts(cfg)


if __name__ == "__main__":
    unittest.main()
