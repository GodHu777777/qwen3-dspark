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
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    prompts = [json.loads(x) for x in Path(args.prompts).read_text().splitlines()]
    if len({x["id"] for x in prompts}) != len(prompts):
        raise ValueError("Duplicate prompt IDs")
    out = Path(args.output)
    (out / "batches").mkdir(parents=True, exist_ok=True)
    model_path = Path(cfg["model"])
    fingerprint = {x.name: digest(x) for x in sorted(model_path.iterdir())
        if x.suffix in {".json", ".safetensors", ".txt", ".jinja"}}
    identity = {"config": cfg, "prompts_sha256": digest(args.prompts),
        "script_sha256": digest(__file__), "model_files_sha256": fingerprint,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "hip": torch.version.hip, "python": platform.python_version(),
        "device": torch.cuda.get_device_name(0),
        "scope": "pilot data regeneration; not a performance benchmark"}
    manifest = out / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError("Resume identity mismatch: use a fresh output directory")
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
        if destination.exists():
            previous = json.loads(destination.read_text())
            if [x["id"] for x in previous] != [x["id"] for x in group]:
                raise ValueError("Batch identity mismatch")
            continue
        seed = cfg["seed"] + start
        set_seed(seed)
        texts = [tokenizer.apply_chat_template(x["messages"], tokenize=False,
            add_generation_prompt=True, enable_thinking=cfg["enable_thinking"]) for x in group]
        inputs = tokenizer(texts, add_special_tokens=False, padding=True, return_tensors="pt").to("cuda")
        for item, mask in zip(group, inputs["attention_mask"]):
            if int(mask.sum()) != item["prompt_tokens"]:
                raise ValueError("Prompt tokenization changed")
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
        print(json.dumps({"done": start + len(group), "total": len(prompts),
            "seconds": round(elapsed, 2), "accepted": sum(x["accepted"] for x in records),
            "tokens": sum(x["generated_tokens"] for x in records)}), flush=True)
    records = [x for path in sorted((out / "batches").glob("*.json")) for x in json.loads(path.read_text())]
    if [x["id"] for x in records] != [x["id"] for x in prompts]:
        raise ValueError("Output population mismatch")
    for name, items in [("records", records), ("rejected", [x for x in records if not x["accepted"]]),
        ("train", [x for x in records if x["accepted"] and x["split"] == "train"]),
        ("validation", [x for x in records if x["accepted"] and x["split"] == "validation"])]:
        target = out / (name + ".jsonl")
        temporary = target.with_suffix(".tmp")
        with temporary.open("w") as f:
            for item in items:
                payload = {"messages": item["messages"], "id": item["id"]} if name in {"train", "validation"} else item
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        os.replace(temporary, target)
    summary = {"total": len(records), "accepted": sum(x["accepted"] for x in records),
        "accepted_splits": dict(Counter(x["split"] for x in records if x["accepted"])),
        "finish_reasons": dict(Counter(x["finish_reason"] for x in records)),
        "rejection_reasons": dict(Counter(r for x in records for r in x["rejection_reasons"])),
        "generated_tokens": sum(x["generated_tokens"] for x in records),
        "training_tokens": sum(x["training_tokens"] for x in records if x["accepted"]),
        "current_invocation_seconds": round(time.monotonic() - run_start, 2),
        "max_gpu_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "output_sha256": {name: digest(out / name) for name in ["train.jsonl", "validation.jsonl", "records.jsonl"]}}
    atomic_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
