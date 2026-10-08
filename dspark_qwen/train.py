"""Single GPU training, held-out teacher-forced evaluation, atomic resume."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed
from .checkpoint import load_checkpoint, save_checkpoint, sha256
from .config import DraftConfig
from .data import load_records, select_anchors, tensors
from .losses import objective
from .model import DSparkDraft
from .target import FrozenTarget


def forward_loss(row, target, draft, generator, device, amp):
    ids, answer_mask = tensors(row, device)
    features = target.capture(ids)
    anchors = select_anchors(answer_mask, draft.spec.num_anchors, generator)
    with torch.autocast("cuda", dtype=torch.bfloat16) if amp else nullcontext():
        output = draft(ids, features.context, anchors)
        # h[t] predicts token[t+1]. Teacher logits are detached and only materialized
        # at selected positions, rather than for the entire target sequence.
        teacher = target.logits(features.last[:, output["label_positions"] - 1])
        return objective(output, teacher, answer_mask, draft.spec)


@torch.no_grad()
def evaluate(rows, target, draft, seed, device, amp):
    draft.eval()
    generator = torch.Generator().manual_seed(seed)
    metrics = []
    for row in rows:
        _, item = forward_loss(row, target, draft, generator, device, amp)
        metrics.append(item)
    draft.train()
    # Explicit macro average across examples, not token-weighted corpus loss.
    return {key: sum(item[key] for item in metrics) / len(metrics) for key in metrics[0]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int, help="Finish at this optimizer step, for bounded/resume checks")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    spec = DraftConfig.from_dict(cfg["draft"])
    out = Path(cfg["output_dir"])
    if out.exists() and not args.resume:
        raise ValueError("Output exists; use a new directory or --resume")
    if args.resume and not (out / "latest").exists():
        raise ValueError("No completed checkpoint to resume")
    out.mkdir(parents=True, exist_ok=True)
    device = cfg.get("device", "cuda")
    amp = device.startswith("cuda")
    set_seed(cfg["seed"])
    generation = json.loads(Path(cfg["generation_manifest"]).read_text())
    source_summary = json.loads((Path(cfg["generation_manifest"]).parent / "summary.json").read_text())
    if sha256(cfg["records"]) != source_summary["output_sha256"]["records.jsonl"]:
        raise ValueError("Data records differ from generation evidence")
    for name, expected in generation["model_files_sha256"].items():
        if sha256(Path(cfg["model"]) / name) != expected:
            raise ValueError(f"Target/tokenizer fingerprint mismatch: {name}")
    identity = {"config": cfg, "records_sha256": sha256(cfg["records"]),
        "target_fingerprint": generation["model_files_sha256"],
        "source_sha256": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "torch": torch.__version__, "transformers": transformers.__version__, "hip": torch.version.hip}
    run_manifest = out / "run.json"
    if args.resume:
        if json.loads(run_manifest.read_text()) != identity:
            raise ValueError("Resume identity mismatch; original run manifest preserved")
    else:
        run_manifest.write_text(json.dumps(identity, indent=2) + "\n")
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], local_files_only=True,
        dtype=torch.bfloat16 if amp else torch.float32, attn_implementation="sdpa").to(device)
    target = FrozenTarget(model, spec)
    draft = DSparkDraft(model, spec).to(device)  # FP32 trainables; frozen target remains BF16.
    parameters = [p for p in draft.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=cfg["learning_rate"], betas=(0.9, 0.95), weight_decay=0.0)
    generator = torch.Generator().manual_seed(cfg["seed"])
    train_rows = load_records(cfg["records"], "train", cfg["max_sequence_tokens"])
    val_rows = load_records(cfg["records"], "validation", cfg["max_sequence_tokens"])
    if {r["id"] for r in train_rows} & {r["id"] for r in val_rows}:
        raise ValueError("Train/validation leakage")
    step = 0
    metadata = {"identity": identity, "draft_config": spec.to_dict(),
                "trainable_parameters": sum(p.numel() for p in parameters),
                "train_rows": len(train_rows), "validation_rows": len(val_rows)}
    if args.resume:
        loaded = load_checkpoint(out / (out / "latest").read_text().strip(), draft, identity, optimizer, generator)
        step = loaded["step"]
        print(json.dumps({"resumed_step": step}), flush=True)
    # Small fixed validation panel, all held-out examples. Same anchor RNG before/after.
    before = evaluate(val_rows, target, draft, cfg["seed"] + 999, device, amp)
    print(json.dumps({"validation_before": before}), flush=True)
    limit = min(cfg["max_steps"], args.stop_after or cfg["max_steps"])
    if limit <= step:
        raise ValueError("Requested stop step must exceed resumed step")
    began = time.monotonic()
    frozen_version = {name: p._version for name, p in model.named_parameters()}
    with (out / "metrics.jsonl").open("a") as log:
        while step < limit:
            optimizer.zero_grad(set_to_none=True)
            items = []
            for micro in range(cfg["gradient_accumulation"]):
                index = (step * cfg["gradient_accumulation"] + micro) % len(train_rows)
                loss, metrics = forward_loss(train_rows[index], target, draft, generator, device, amp)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite loss")
                (loss / cfg["gradient_accumulation"]).backward()
                items.append(metrics)
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, cfg["max_grad_norm"], error_if_nonfinite=True)
            # All components should learn; check first update rather than just total loss.
            if step == 0:
                groups = {"projection": draft.fc, "backbone": draft.layers,
                          "markov": draft.markov_projection, "confidence": draft.confidence}
                gradient_checks = {name: any(p.grad is not None and bool(p.grad.abs().sum() > 0)
                    for p in module.parameters()) for name, module in groups.items()}
                if not all(gradient_checks.values()):
                    raise RuntimeError(f"Missing gradient path: {gradient_checks}")
                old_projection = draft.fc.weight.detach().clone()
            optimizer.step()
            if step == 0 and torch.equal(old_projection, draft.fc.weight):
                raise RuntimeError("Optimizer did not change projection weights")
            step += 1
            row = {"step": step, **{k: sum(x[k] for x in items) / len(items) for k in items[0]},
                   "gradient_norm": float(grad_norm), "elapsed_seconds": time.monotonic() - began}
            log.write(json.dumps(row) + "\n"); log.flush()
            print(json.dumps(row), flush=True)
    after = evaluate(val_rows, target, draft, cfg["seed"] + 999, device, amp)
    frozen_ok = all(p.grad is None and p._version == frozen_version[name] for name, p in model.named_parameters())
    if not frozen_ok:
        raise RuntimeError("Frozen target was modified or acquired gradients")
    ckpt = save_checkpoint(out, step, draft, optimizer, generator, metadata)
    report = {"step": step, "checkpoint": str(ckpt), "validation_before": before,
        "validation_after": after, "target_frozen_verified": frozen_ok,
        "trainable_parameters": metadata["trainable_parameters"], "train_rows": len(train_rows),
        "validation_rows": len(val_rows), "amp": "bfloat16" if amp else "none",
        "trainable_dtype": "float32", "elapsed_seconds": time.monotonic() - began,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if amp else None,
        "scope": "short training execution check; teacher-forced metrics are not rollout acceptance or speedup"}
    (out / f"result-step-{step:06d}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
