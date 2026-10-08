"""Regenerate pilot answers; preserve exact tokens and quarantine truncations."""
import argparse
import hashlib
import json
import os
import platform
import time
from collections import Counter
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

try:
    from .data_pipeline import export_records, validate_prompts, completed_batch
except ImportError:
    from data_pipeline import export_records, validate_prompts, completed_batch


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--prompts", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--source-commit", help="Committed source deployed for this generation run")
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    prompts = [json.loads(x) for x in Path(args.prompts).read_text().splitlines()]
    selection_path = Path(args.prompts).with_name("selection.json")
    selection = json.loads(selection_path.read_text())
    if selection["config"] != cfg or selection["prompts_sha256"] != digest(args.prompts):
        raise ValueError("Selection configuration or prompt bytes changed")
    if "identity" in selection and selection["identity"]["config_sha256"] != digest(args.config):
        raise ValueError("Configuration bytes differ from immutable selection")
    exclusion_path = Path(args.prompts).with_name("excluded-prompt-ids.json")
    excluded = json.loads(exclusion_path.read_text()) if exclusion_path.exists() else []
    if "output_sha256" in selection:
        for name, expected in selection["output_sha256"].items():
            if digest(Path(args.prompts).parent / name) != expected:
                raise ValueError("Selection artifact hash mismatch")
    if cfg.get("require_exclusions") and not excluded:
        raise ValueError("Expansion requires prior-prompt exclusions")
    validate_prompts(prompts, cfg, excluded)
    out = Path(args.output)
    (out / "batches").mkdir(parents=True, exist_ok=True, mode=0o700)
    out.chmod(0o700)
    model_path = Path(cfg["model"])
    fingerprint = {x.name: digest(x) for x in sorted(model_path.iterdir())
        if x.suffix in {".json", ".safetensors", ".txt", ".jinja"}}
    identity = {"config": cfg, "config_sha256": digest(args.config),
        "source_commit": args.source_commit,
        "selection_sha256": digest(selection_path), "prompts_sha256": digest(args.prompts),
        "pipeline_sha256": digest(Path(__file__).with_name("data_pipeline.py")),
        "script_sha256": digest(__file__), "model_files_sha256": fingerprint,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "hip": torch.version.hip, "python": platform.python_version(),
        "device": torch.cuda.get_device_name(0),
        "scope": "target data regeneration; final test excluded from training records; not a benchmark"}
    if cfg.get("require_exclusions") and (not args.source_commit or len(args.source_commit) != 40
            or any(c not in "0123456789abcdef" for c in args.source_commit)):
        raise ValueError("Expansion generation requires the full committed --source-commit SHA")
    manifest = out / "manifest.json"
    if not manifest.exists() and any((out / "batches").iterdir()):
        raise ValueError("Unmanifested batch artifacts: use a fresh output directory")
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError("Resume identity mismatch: use a fresh output directory")
    if "identity" in selection:
        for name, expected in selection["identity"]["tokenizer_files_sha256"].items():
            if fingerprint.get(name) != expected:
                raise ValueError("Tokenizer/model configuration changed after selection")
    atomic_json(manifest, identity)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Save the original tokenizer; enable_thinking=False remains an explicit requirement.
    tokenizer.save_pretrained(out / "tokenizer")
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
        dtype=getattr(torch, cfg["dtype"]), attn_implementation=cfg["attention_backend"]).to("cuda").eval()
    eos_ids = model.generation_config.eos_token_id
    eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
    run_start = time.monotonic()
    for start in range(0, len(prompts), cfg["batch_size"]):
        destination = out / "batches" / f"{start:06d}.json"
        group = prompts[start:start + cfg["batch_size"]]
        checksum = destination.with_suffix(".sha256")
        if completed_batch(destination, [x["id"] for x in group]) is not None:
            continue
        seed = cfg["seed"] + start
        set_seed(seed)
        texts = [tokenizer.apply_chat_template(x["messages"], tokenize=False,
            add_generation_prompt=True, enable_thinking=cfg["enable_thinking"]) for x in group]
        inputs = tokenizer(texts, add_special_tokens=False, padding=True, return_tensors="pt").to("cuda")
        for i, (item, mask) in enumerate(zip(group, inputs["attention_mask"])):
            if int(mask.sum()) != item["prompt_tokens"]:
                raise ValueError("Prompt tokenization changed")
            if "prompt_token_ids" in item and inputs["input_ids"][i][mask.bool()].tolist() != item["prompt_token_ids"]:
                raise ValueError("Exact prompt token IDs changed")
        torch.cuda.synchronize()
        began = time.monotonic()
        with torch.inference_mode():
            outputs = model.generate(**inputs, do_sample=True,
                temperature=cfg["temperature"], top_p=cfg["top_p"], top_k=cfg["top_k"],
                min_p=cfg["min_p"], repetition_penalty=1.0,
                max_new_tokens=cfg["max_new_tokens"],
                pad_token_id=tokenizer.pad_token_id, use_cache=True)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - began
        records = []
        for i, item in enumerate(group):
            ids = outputs[i, inputs["input_ids"].shape[1]:].tolist()
            stop = next((j for j, t in enumerate(ids) if t in eos_ids), None)
            finish = "eos" if stop is not None else "length"
            if stop is not None:
                ids = ids[:stop + 1]
            content = tokenizer.decode(ids, skip_special_tokens=True)
            prompt_ids = inputs["input_ids"][i][inputs["attention_mask"][i].bool()].tolist()
            if "prompt_token_ids" in item and prompt_ids != item["prompt_token_ids"]:
                raise ValueError("Exact prompt token IDs changed")
            messages = item["messages"] + [{"role": "assistant", "content": content}]
            rendered = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
                add_generation_prompt=False, enable_thinking=cfg["enable_thinking"])
            reasons = []
            if finish != "eos": reasons.append("truncated")
            if not content.strip(): reasons.append("empty")
            if len(rendered) > cfg["max_sequence_tokens"]: reasons.append("sequence_too_long")
            if "<think>" in content or "</think>" in content: reasons.append("unexpected_thinking_tags")
            # A template may append a final newline after EOS; preserve both forms for audit.
            generated = prompt_ids + ids
            match = rendered[:len(generated)] == generated
            if finish == "eos" and not match: reasons.append("template_token_mismatch")
            records.append({**item, "messages": messages, "finish_reason": finish,
                "generated_tokens": len(ids), "training_tokens": len(rendered),
                "prompt_token_ids": prompt_ids, "output_token_ids": ids,
                "batch_seed": seed, "batch_seconds": elapsed,
                "template_prefix_matches_generation": match,
                "accepted": not reasons, "rejection_reasons": reasons})
        atomic_json(destination, records)
        temporary_checksum = checksum.with_suffix(".tmp")
        temporary_checksum.write_text(digest(destination) + "\n")
        os.replace(temporary_checksum, checksum)
        print(json.dumps({"done": start + len(group), "total": len(prompts),
            "seconds": round(elapsed, 2), "accepted": sum(x["accepted"] for x in records),
            "tokens": sum(x["generated_tokens"] for x in records)}), flush=True)
    records = [x for path in sorted((out / "batches").glob("*.json")) for x in json.loads(path.read_text())]
    if [x["id"] for x in records] != [x["id"] for x in prompts]:
        raise ValueError("Output population mismatch")
    output_hashes = export_records(out, records, cfg)
    summary = {"total": len(records), "accepted": sum(x["accepted"] for x in records),
        "accepted_splits": dict(Counter(x["split"] for x in records if x["accepted"])),
        "finish_reasons": dict(Counter(x["finish_reason"] for x in records)),
        "rejection_reasons": dict(Counter(r for x in records for r in x["rejection_reasons"])),
        "generated_tokens": sum(x["generated_tokens"] for x in records),
        "training_tokens": sum(x["training_tokens"] for x in records if x["accepted"]),
        "current_invocation_seconds": round(time.monotonic() - run_start, 2),
        "max_gpu_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "eos_token_ids": sorted(eos_ids),
        "final_test_locked": bool(cfg.get("test_samples", 0)),
        "output_sha256": output_hashes}
    atomic_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
