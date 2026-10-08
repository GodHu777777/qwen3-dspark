import json
from pathlib import Path
import tempfile
import unittest

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from dspark_qwen.config import DraftConfig
from dspark_qwen.eval_dev_tf import evaluate_panel, fixed_panel
from dspark_qwen.experiment_inputs import development_rows
from dspark_qwen.memory_gate import exercise_step, pareto_training_rows
from dspark_qwen.model import DSparkDraft
from dspark_qwen.target import FrozenTarget


def record(identity, split, prompt, output, accepted=True):
    return {"id": identity, "split": split, "prompt_token_ids": prompt,
        "output_token_ids": output, "accepted": accepted,
        "template_prefix_matches_generation": True, "finish_reason": "eos"}


class TrainingToolTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)
        config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, max_position_embeddings=256, attention_dropout=0.0)
        config._attn_implementation = "sdpa"
        self.model = Qwen3ForCausalLM(config).eval()
        self.spec = DraftConfig(layer_ids=(0, 2), num_layers=2, block_size=7,
            markov_rank=8, mask_token_id=63, num_anchors=32)
        self.target = FrozenTarget(self.model, self.spec)
        self.draft = DSparkDraft(self.model, self.spec)

    def write_records(self, directory, records):
        path = Path(directory) / "records.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in records))
        return path

    def test_memory_gate_uses_longest_eligible_train_and_allocates_adam_states(self):
        records = [record("short", "train", [1, 2], [3, 62]),
            record("longest", "train", [1, 2, 3], list(range(4, 45)) + [62]),
            record("validation-long", "validation", [1], [2] * 60 + [62]),
            record("rejected-long", "train", [1], [2] * 60 + [62], accepted=False),
            record("over-budget", "train", [1], [2] * 90 + [62])]
        with tempfile.TemporaryDirectory() as d:
            path = self.write_records(d, records)
            rows = development_rows(path, "train", 64)
        cfg = {"gradient_accumulation": 2, "learning_rate": 6e-4, "seed": 37, "max_grad_norm": 1.0}
        projection = self.draft.fc.weight.detach().clone()
        report = exercise_step(rows, self.target, self.draft, cfg, "cpu")
        self.assertEqual(report["record_id"], "longest")
        self.assertEqual(report["sequence_tokens"], 45)
        self.assertEqual(report["actual_anchors_per_microstep"], 32)
        self.assertEqual(len(report["micro_metrics"]), 4)
        self.assertTrue(report["target_frozen_verified"])
        self.assertTrue(report["first_adam_update_verified"])
        self.assertTrue(report["state_resident_cycle_verified"])
        self.assertEqual(report["cycles"][0]["optimizer_state_bytes_before_cycle"], 0)
        self.assertGreater(report["cycles"][1]["optimizer_state_bytes_before_cycle"], 0)
        self.assertEqual([len(c["micro_metrics"]) for c in report["cycles"]], [2, 2])
        self.assertGreater(report["optimizer_state_tensor_bytes"], 0)
        self.assertTrue(all(report["gradient_checks"].values()))
        self.assertFalse(torch.equal(projection, self.draft.fc.weight))
        self.assertIsNone(report["memory_after_first_adam_update"])

    def test_validation_panel_rejects_any_test_or_split_identity_leak(self):
        valid = record("valid", "validation", [1], [2, 62])
        with tempfile.TemporaryDirectory() as d:
            for other in (record("test", "test", [1], [62]),
                          record("test-rejected", "test", [1], [62], accepted=False),
                          record("valid", "train", [1], [62])):
                path = self.write_records(d, [valid, other])
                with self.assertRaises(ValueError):
                    development_rows(path, "validation", 64)
            path = self.write_records(d, [valid])
            with self.assertRaises(ValueError): development_rows(path, "test", 64)

    def test_memory_shape_frontier_covers_shorter_records_with_more_anchors(self):
        rows = [{"id": "long-short-answer", "input_ids": [1] * 60, "answer_start": 59},
            {"id": "medium", "input_ids": [1] * 55, "answer_start": 45},
            {"id": "short-full-anchors", "input_ids": [1] * 50, "answer_start": 10},
            {"id": "dominated", "input_ids": [1] * 40, "answer_start": 20}]
        frontier = pareto_training_rows(rows, 32)
        self.assertEqual([r["id"] for r in frontier], ["long-short-answer", "medium", "short-full-anchors"])
        rows[0]["answer_start"] = 20
        self.assertEqual([r["id"] for r in pareto_training_rows(rows, 32)], ["long-short-answer"])

    def test_fixed_validation_eos_tail_positions_and_repeatability(self):
        records = [record("train", "train", [1], [2, 62]),
            record("one-eos", "validation", [1], [62]),
            record("three-tokens", "validation", [1, 2], [3, 4, 62])]
        with tempfile.TemporaryDirectory() as d:
            rows = development_rows(self.write_records(d, records), "validation", 64)
        panel = fixed_panel(rows, 32)
        self.assertEqual([r["id"] for r in panel], ["one-eos", "three-tokens"])
        self.assertEqual(len(fixed_panel(rows, 1)), 1)
        before = self.draft.fc.weight.detach().clone()
        first = evaluate_panel(panel, self.target, self.draft, 41, "cpu")
        second = evaluate_panel(panel, self.target, self.draft, 41, "cpu")
        self.assertEqual(first["panel_sha256"], second["panel_sha256"])
        self.assertEqual(first["macro_metrics"], second["macro_metrics"])
        # All anchors selected: the one-EOS example has one target; the three
        # completion tokens contribute [3,2,1] valid positions, including EOS.
        self.assertEqual(first["position_denominators"], [4, 2, 1, 0, 0, 0, 0])
        self.assertEqual(first["unweighted_by_position"]["teacher_top1"][3:], [None] * 4)
        self.assertEqual(first["panel"]["anchors_by_id"]["one-eos"], [0])
        self.assertEqual(first["panel"]["anchors_by_id"]["three-tokens"], [1, 2, 3])
        self.assertEqual(first["rollout_numerical_gate"]["status"], "not_run")
        self.assertFalse(first["rollout_numerical_gate"]["checkpoint_selection_by_rollout_allowed"])
        torch.testing.assert_close(before, self.draft.fc.weight, rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        self.assertTrue(all(p.grad is None for p in self.draft.parameters()))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
