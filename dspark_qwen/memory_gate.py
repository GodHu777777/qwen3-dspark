"""Exercise real Pareto shapes, first AdamW allocation and state-resident training.

This writes no checkpoint and does not warm-start a later run. It measures the
chosen dataset/config on current hardware; shorter diagnostic sequences cannot
substitute for this resource gate.
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, set_seed

from .config import DraftConfig
from .experiment_inputs import development_rows, verify_identity
from .model import DSparkDraft
from .target import FrozenTarget
from .train import forward_loss


def memory_snapshot(device):
    if not str(device).startswith("cuda"):
        return None
    torch.cuda.synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    return {"free_bytes": free, "total_bytes": total,
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}


def pareto_training_rows(rows, requested_anchors):
    """Real shapes not dominated in both sequence length and actual anchors."""
    if not rows or type(requested_anchors) is not int or requested_anchors < 1:
        raise ValueError("Need training rows and a positive anchor count")
    ordered = sorted(rows, key=lambda row: (-len(row["input_ids"]),
        -min(requested_anchors, len(row["input_ids"]) - row["answer_start"]), row["id"]))
    frontier, maximum_anchors = [], -1
    for row in ordered:
        actual = min(requested_anchors, len(row["input_ids"]) - row["answer_start"])
        if actual > maximum_anchors:
            frontier.append(row)
            maximum_anchors = actual
    return frontier


def exercise_step(rows, target, draft, cfg, device, probe_row=None, cycle_reports=None):
    """CPU-testable two-cycle update on a selected real training shape."""
    if not rows:
        raise ValueError("No training records for memory gate")
    row = probe_row if probe_row is not None else max(rows, key=lambda r: (len(r["input_ids"]), r["id"]))
    if row not in rows:
        raise ValueError("Probe is not an eligible training record")
    accumulation = cfg["gradient_accumulation"]
    if type(accumulation) is not int or accumulation < 1:
        raise ValueError("gradient_accumulation must be positive")
    parameters = [p for p in draft.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=cfg["learning_rate"], betas=(0.9, 0.95), weight_decay=0.0)
    versions = {name: p._version for name, p in target.model.named_parameters()}
    generator = torch.Generator().manual_seed(cfg["seed"])
    amp = str(device).startswith("cuda")
    draft.train()
    cycles = cycle_reports if cycle_reports is not None else []
    micro_metrics = []
    for cycle in range(2):
        optimizer.zero_grad(set_to_none=True)
        if amp:
            torch.cuda.reset_peak_memory_stats(device)
        resident_bytes_before = sum(value.numel() * value.element_size()
            for state in optimizer.state.values() for value in state.values() if torch.is_tensor(value))
        memory_before_cycle = memory_snapshot(device)
        cycle_metrics = []
        for _ in range(accumulation):
            loss, metrics = forward_loss(row, target, draft, generator, device, amp)
            if not torch.isfinite(loss):
                raise FloatingPointError("Memory-gate loss is nonfinite")
            (loss / accumulation).backward()
            cycle_metrics.append(metrics)
            micro_metrics.append(metrics)
        norm = torch.nn.utils.clip_grad_norm_(parameters, cfg["max_grad_norm"], error_if_nonfinite=True)
        groups = {"projection": draft.fc, "backbone": draft.layers,
            "markov_embedding": draft.markov_embedding, "markov_projection": draft.markov_projection,
            "confidence": draft.confidence}
        gradients = {name: any(p.grad is not None and bool(p.grad.abs().sum() > 0)
            for p in module.parameters()) for name, module in groups.items()}
        if not all(gradients.values()):
            raise RuntimeError(f"Missing gradient path: {gradients}")
        old_projection = draft.fc.weight.detach().clone()
        before_update = memory_snapshot(device)
        optimizer.step()
        if torch.equal(old_projection, draft.fc.weight):
            raise RuntimeError(f"Adam update {cycle + 1} did not change projection")
        del old_projection
        cycles.append({"cycle": cycle + 1,
            "purpose": "first_adam_allocation" if cycle == 0 else "adam_state_resident_training",
            "optimizer_state_bytes_before_cycle": resident_bytes_before,
            "micro_metrics": cycle_metrics, "gradient_norm": float(norm), "gradient_checks": gradients,
            "memory_before_cycle": memory_before_cycle, "memory_before_update": before_update,
            "memory_after_update": memory_snapshot(device)})
    frozen = all(p.grad is None and p._version == versions[name] for name, p in target.model.named_parameters())
    if not frozen:
        raise RuntimeError("Frozen target was modified or acquired gradients")
    state_tensors = [value for state in optimizer.state.values() for value in state.values()
        if torch.is_tensor(value)]
    return {"passed": True, "record_id": row["id"], "training_records": len(rows),
        "sequence_tokens": len(row["input_ids"]), "completion_tokens": len(row["input_ids"]) - row["answer_start"],
        "longest_sequence_verified": len(row["input_ids"]) == max(len(r["input_ids"]) for r in rows),
        "requested_anchors": draft.spec.num_anchors,
        "actual_anchors_per_microstep": min(draft.spec.num_anchors, len(row["input_ids"]) - row["answer_start"]),
        "gradient_accumulation": accumulation, "micro_metrics": micro_metrics,
        "gradient_norm": float(norm), "gradient_checks": gradients, "target_frozen_verified": frozen,
        "first_adam_update_verified": True, "state_resident_cycle_verified": True, "cycles": cycles,
        "optimizer_state_tensor_bytes": sum(t.numel() * t.element_size() for t in state_tensors),
        "memory_before_first_adam_update": cycles[0]["memory_before_update"],
        "memory_after_first_adam_update": cycles[0]["memory_after_update"],
        "peak_allocated_across_cycles": max(c["memory_after_update"]["peak_allocated_bytes"] for c in cycles) if amp else None,
        "peak_reserved_across_cycles": max(c["memory_after_update"]["peak_reserved_bytes"] for c in cycles) if amp else None,
        "trainable_parameters": sum(p.numel() for p in parameters)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    # Check data and reject final test before loading a model or touching CUDA.
    rows = development_rows(cfg["records"], "train", cfg["max_sequence_tokens"])
    identity = verify_identity(cfg)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    (out / "run.json").write_text(json.dumps(identity, indent=2) + "\n")
    device = cfg.get("device", "cuda")
    amp = device.startswith("cuda")
    report = {"identity": identity, "scope": "real Pareto training shapes, first allocation and state-resident accumulation; empirical resource gate, not a formal worst-case bound",
        "probes": []}
    try:
        if amp:
            torch.cuda.reset_peak_memory_stats(device)
        report["device"] = str(device)
        report["device_name"] = torch.cuda.get_device_name(device) if amp else "CPU"
        report["target_dtype"] = "bfloat16" if amp else "float32"
        report["trainable_dtype"] = "float32"
        report["memory_before_model"] = memory_snapshot(device)
        set_seed(cfg["seed"])
        model = AutoModelForCausalLM.from_pretrained(cfg["model"], local_files_only=True,
            dtype=torch.bfloat16 if amp else torch.float32, attn_implementation="sdpa").to(device)
        spec = DraftConfig.from_dict(cfg["draft"])
        target = FrozenTarget(model, spec)
        probes = pareto_training_rows(rows, spec.num_anchors)
        report["shape_selection"] = {"training_records": len(rows), "probe_count": len(probes),
            "max_sequence_tokens": max(len(r["input_ids"]) for r in rows),
            "max_actual_anchors": max(min(spec.num_anchors, len(r["input_ids"]) - r["answer_start"]) for r in rows),
            "dimensions": ["sequence_tokens", "actual_anchors"],
            "limitation": "Measured real nondominated shapes; kernel workspace and allocator behavior need not be monotone"}
        for row in probes:
            set_seed(cfg["seed"])
            draft = DSparkDraft(model, spec).to(device)
            report["memory_after_model"] = memory_snapshot(device)
            report["active_probe"] = {"record_id": row["id"], "sequence_tokens": len(row["input_ids"]),
                "actual_anchors": min(spec.num_anchors, len(row["input_ids"]) - row["answer_start"]),
                "completed_cycles": []}
            report["probes"].append(exercise_step(rows, target, draft, cfg, device, probe_row=row,
                cycle_reports=report["active_probe"]["completed_cycles"]))
            del report["active_probe"]
            del draft
            if amp:
                torch.cuda.empty_cache()
        report["passed"] = all(p["passed"] for p in report["probes"])
        report["target_frozen_verified"] = all(p["target_frozen_verified"] for p in report["probes"])
        report["peak_allocated_across_probes"] = max(p["peak_allocated_across_cycles"] for p in report["probes"]) if amp else None
    except Exception as exc:
        report.update(passed=False, error_type=type(exc).__name__, error=str(exc))
        try:
            report["memory_at_failure"] = memory_snapshot(device)
        except Exception:
            report["memory_at_failure"] = None
        raise
    finally:
        (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
