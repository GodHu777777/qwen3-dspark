"""Bounded single-record overfit diagnostic, never a generalization benchmark.

Cache frozen target features for one record and train on fixed anchors. Compare
teacher-forced agreement and actual greedy rollout before/after learning.
"""
import argparse
import json
from pathlib import Path
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed

from .checkpoint import save_checkpoint, sha256
from .config import DraftConfig
from .data import load_records, tensors
from .decode import speculative_greedy, target_greedy
from .losses import objective
from .model import DSparkDraft
from .target import FrozenTarget


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--record-index", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=32)
    parser.add_argument("--rollout-tokens", type=int, default=32)
    args = parser.parse_args()
    if min(args.steps, args.eval_every, args.rollout_tokens) < 1 or args.record_index < 0:
        parser.error("steps, eval-every and rollout-tokens must be positive; index must be nonnegative")
    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    spec = DraftConfig.from_dict(cfg["draft"])
    generation = json.loads(Path(cfg["generation_manifest"]).read_text())
    summary = json.loads((Path(cfg["generation_manifest"]).parent / "summary.json").read_text())
    if sha256(cfg["records"]) != summary["output_sha256"]["records.jsonl"]:
        raise ValueError("Data records differ from generation evidence")
    for name, digest in generation["model_files_sha256"].items():
        if sha256(Path(cfg["model"]) / name) != digest:
            raise ValueError(f"Target fingerprint mismatch: {name}")
    identity = {"config": cfg, "diagnostic_args": vars(args),
        "records_sha256": sha256(cfg["records"]), "target_fingerprint": generation["model_files_sha256"],
        "source_sha256": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "torch": torch.__version__, "transformers": transformers.__version__, "hip": torch.version.hip}
    (out / "run.json").write_text(json.dumps(identity, indent=2) + "\n")
    set_seed(cfg["seed"])
    rows = load_records(cfg["records"], "train", cfg["max_sequence_tokens"])
    row = rows[args.record_index]
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    target = FrozenTarget(model, spec)
    draft = DSparkDraft(model, spec).to("cuda")
    parameters = [p for p in draft.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=cfg["learning_rate"], betas=(0.9, 0.95), weight_decay=0.0)
    generator = torch.Generator().manual_seed(cfg["seed"])
    ids, answer_mask = tensors(row, "cuda")
    # First block explicitly covers the rollout start; subsequent anchors cover
    # the next blocks so this is a meaningful bounded learning-capacity check.
    anchors = torch.arange(row["answer_start"] - 1,
        min(ids.shape[1] - 1, row["answer_start"] - 1 + spec.num_anchors * spec.block_size),
        spec.block_size, device="cuda")
    features = target.capture(ids)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        initial = draft(ids, features.context, anchors)
        teacher = target.logits(features.last[:, initial["label_positions"] - 1]).detach()
    del initial
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    prompt = ids[:, :row["answer_start"]]
    baseline = target_greedy(target, prompt, args.rollout_tokens, eos)
    frozen_versions = {name: p._version for name, p in model.named_parameters()}
    start = time.monotonic()

    @torch.no_grad()
    def evaluate(step):
        draft.eval()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = draft(ids, features.context, anchors)
            _, metrics = objective(output, teacher, answer_mask, spec)
        valid = output["valid"] & answer_mask[:, output["label_positions"]]
        match = output["logits"].argmax(-1) == teacher.argmax(-1)
        metrics["teacher_top1_agreement"] = match[valid].float().mean().item()
        metrics["label_top1_accuracy"] = (output["logits"].argmax(-1) == output["labels"])[valid].float().mean().item()
        metrics["teacher_top1_by_position"] = [match[..., k][valid[..., k]].float().mean().item()
            if valid[..., k].any() else None for k in range(spec.block_size)]
        actual, rounds = speculative_greedy(target, draft, prompt, args.rollout_tokens, eos, amp=True)
        if actual != baseline:
            raise RuntimeError("Greedy equivalence failure")
        metrics.update(step=step, elapsed_seconds=time.monotonic() - start,
            rollout_equal=True, rollout_tokens=actual, rounds=rounds,
            accepted_per_round=sum(r["accepted"] for r in rounds) / max(1, len(rounds)))
        draft.train()
        print(json.dumps({"evaluation": metrics}), flush=True)
        with (out / "evaluation.jsonl").open("a") as f:
            f.write(json.dumps(metrics) + "\n")

    evaluate(0)
    with (out / "metrics.jsonl").open("w") as log:
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = draft(ids, features.context, anchors)
                loss, metrics = objective(output, teacher, answer_mask, spec)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, cfg["max_grad_norm"], error_if_nonfinite=True)
            optimizer.step()
            metrics.update(step=step, gradient_norm=float(norm), elapsed_seconds=time.monotonic() - start)
            log.write(json.dumps(metrics) + "\n"); log.flush()
            if step % 8 == 0:
                print(json.dumps(metrics), flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                evaluate(step)
    frozen_ok = all(p.grad is None and p._version == frozen_versions[name] for name, p in model.named_parameters())
    if not frozen_ok:
        raise RuntimeError("Frozen target changed")
    metadata = {"identity": identity, "draft_config": spec.to_dict(), "record_id": row["id"],
        "anchors": anchors.tolist(), "trainable_parameters": sum(p.numel() for p in parameters),
        "target_frozen_verified": frozen_ok, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "scope": "single-record fixed-anchor overfit; not held-out quality or speed evidence"}
    cp = save_checkpoint(out, args.steps, draft, optimizer, generator, metadata)
    (out / "result.json").write_text(json.dumps({**metadata, "checkpoint": str(cp),
        "elapsed_seconds": time.monotonic() - start}, indent=2) + "\n")


if __name__ == "__main__":
    main()
