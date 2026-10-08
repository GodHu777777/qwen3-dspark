"""Fixed validation-only teacher-forced checkpoint evaluation.

These are surrogate quality diagnostics, not rollout acceptance, lossless decode,
confidence calibration on generated prefixes, or speed measurements. Final-test
records are rejected even if a caller attempts to select only validation rows.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from .checkpoint import load_checkpoint, sha256
from .config import DraftConfig
from .data import select_anchors, tensors
from .experiment_inputs import development_rows, verify_identity
from .losses import objective
from .model import DSparkDraft
from .target import FrozenTarget


def fixed_panel(rows, count):
    if type(count) is not int or count < 1:
        raise ValueError("Panel size must be a positive integer")
    if not rows:
        raise ValueError("No validation records")
    panel = rows[:count]  # Frozen records order, never chosen by model results.
    if len({row["id"] for row in panel}) != len(panel):
        raise ValueError("Duplicate panel identities")
    return panel


@torch.no_grad()
def evaluate_panel(rows, target, draft, seed, device, amp=False):
    if not rows:
        raise ValueError("No validation panel records")
    draft.eval()
    generator = torch.Generator().manual_seed(seed)
    k = draft.spec.block_size
    totals = {name: torch.zeros(k, dtype=torch.float64) for name in (
        "valid", "teacher_top1", "label_top1", "overlap", "confidence", "confidence_abs_error")}
    examples, anchors_by_id = [], {}
    for row in rows:
        ids, answer_mask = tensors(row, device)
        anchors = select_anchors(answer_mask, draft.spec.num_anchors, generator)
        features = target.capture(ids)
        with torch.autocast("cuda", dtype=torch.bfloat16) if amp else nullcontext():
            output = draft(ids, features.context, anchors)
            teacher = target.logits(features.last[:, output["label_positions"] - 1])
            loss, metrics = objective(output, teacher, answer_mask, draft.spec)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite validation objective")
        valid = output["valid"] & answer_mask[:, output["label_positions"]]
        predicted = output["logits"].argmax(-1)
        teacher_top1 = (predicted == teacher.argmax(-1)).float()
        label_top1 = (predicted == output["labels"]).float()
        overlap = (1 - 0.5 * (teacher.float().softmax(-1) - output["logits"].float().softmax(-1)).abs().sum(-1)).clamp(0, 1)
        confidence = output["confidence"].float().sigmoid()
        values = {"valid": torch.ones_like(overlap), "teacher_top1": teacher_top1,
            "label_top1": label_top1, "overlap": overlap, "confidence": confidence,
            "confidence_abs_error": (confidence - overlap).abs()}
        for name, value in values.items():
            totals[name] += (value * valid).sum(dim=(0, 1)).double().cpu()
        anchors_by_id[row["id"]] = anchors.tolist()
        examples.append({"id": row["id"], **metrics,
            "teacher_top1": teacher_top1[valid].mean().item(),
            "label_top1": label_top1[valid].mean().item(),
            "sequence_tokens": ids.shape[1], "anchor_count": anchors.numel()})
    denominator = totals["valid"]
    by_position = {name: [float(n / d) if d else None for n, d in zip(value, denominator)]
        for name, value in totals.items() if name != "valid"}
    metric_keys = [key for key in examples[0] if key not in ("id", "sequence_tokens", "anchor_count")]
    macro = {key: sum(example[key] for example in examples) / len(examples) for key in metric_keys}
    panel_description = {"ids": [r["id"] for r in rows], "anchor_seed": seed,
        "anchors_by_id": anchors_by_id, "block_size": k}
    panel_hash = hashlib.sha256(json.dumps(panel_description, sort_keys=True).encode()).hexdigest()
    return {"panel": panel_description, "panel_sha256": panel_hash, "example_count": len(rows),
        "macro_metrics": macro, "position_denominators": denominator.long().tolist(),
        "unweighted_by_position": by_position,
        "first_position_teacher_top1": by_position["teacher_top1"][0],
        "zero_confidence_baseline_mae_by_position": by_position["overlap"],
        "examples": examples,
        "rollout_numerical_gate": {"status": "not_run", "checkpoint_selection_by_rollout_allowed": False,
            "reason": "Teacher-forced evaluation does not validate cached decoding or actual acceptance"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--panel-size", type=int, default=32)
    parser.add_argument("--anchor-seed", type=int, default=20262007)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    cp = Path(args.checkpoint)
    metadata = json.loads((cp / "metadata.json").read_text())
    cfg = metadata["identity"]["config"]
    rows = development_rows(cfg["records"], "validation", cfg["max_sequence_tokens"])
    panel = fixed_panel(rows, args.panel_size)
    identity = verify_identity(cfg, expected=metadata["identity"])
    identity.update(checkpoint_weights_sha256=sha256(cp / "draft.safetensors"),
        checkpoint_metadata_sha256=sha256(cp / "metadata.json"), evaluation_args=vars(args))
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    (out / "run.json").write_text(json.dumps(identity, indent=2) + "\n")
    amp = args.device.startswith("cuda")
    report = {"identity": identity, "scope": "validation-only fixed-panel teacher-forced metrics; not rollout quality or speed",
        "available_validation_records": len(rows), "requested_panel_size": args.panel_size,
        "target_dtype": "bfloat16" if amp else "float32"}
    try:
        model = AutoModelForCausalLM.from_pretrained(cfg["model"], local_files_only=True,
            dtype=torch.bfloat16 if amp else torch.float32, attn_implementation="sdpa").to(args.device)
        spec = DraftConfig.from_dict(metadata["draft_config"])
        target, draft = FrozenTarget(model, spec), DSparkDraft(model, spec).to(args.device)
        load_checkpoint(cp, draft)
        report.update(evaluate_panel(panel, target, draft, args.anchor_seed, args.device, amp))
        report["passed"] = True
    except Exception as exc:
        report.update(passed=False, error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({key: report[key] for key in ("passed", "macro_metrics", "first_position_teacher_top1")
            if key in report}), flush=True)


if __name__ == "__main__":
    main()
