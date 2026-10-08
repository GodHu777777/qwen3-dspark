"""Bounded real-case target-only probe. Private token-bearing output."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from transformers import AutoModelForCausalLM
from prototype import FixedTarget, capture_shapes
from dspark_qwen.cached_target import CachedTarget
from dspark_qwen.config import DraftConfig
from dspark_qwen.target import FrozenTarget
from dspark_qwen.eval_cached_decode import diagnose_same_prefix


def top(logits):
    values, tokens = logits.float().topk(5)
    return {"argmax": int(logits.argmax()), "values": values.flatten().tolist(), "tokens": tokens.flatten().tolist()}


@torch.no_grad()
def canonical_sequence(model, ids, capacity, count, eos):
    target = FixedTarget(model, capacity)
    logits = target.prefill(ids)[:, -1]
    tokens, values = [int(logits.argmax())], [logits.clone()]
    while len(tokens) < count and tokens[-1] not in eos:
        before = target.length
        logits = target.query(ids.new_tensor([[tokens[-1]]]))[:, 0]
        target.crop(before + 1)
        tokens.append(int(logits.argmax()))
        values.append(logits.clone())
    return tokens, values


@torch.no_grad()
def canonical_oracle_block(model, ids, capacity, reference, count, eos):
    target = FixedTarget(model, capacity)
    first = int(target.prefill(ids)[:, -1].argmax())
    tokens = [first]
    while len(tokens) < min(count, len(reference)) and tokens[-1] not in eos:
        before, offset = target.length, len(tokens)
        proposals = reference[offset:offset + 7]
        logits = target.query(ids.new_tensor([[tokens[-1]] + proposals]))[0]
        predictions = logits.argmax(-1).tolist()
        committed = []
        for proposed, predicted in zip(proposals, predictions):
            committed.append(predicted)
            if proposed != predicted or predicted in eos:
                break
        else:
            committed.append(predictions[len(proposals)])
        committed = committed[:count - len(tokens)]
        eos_index = next((i for i, t in enumerate(committed) if t in eos), None)
        if eos_index is not None:
            committed = committed[:eos_index + 1]
        tokens.extend(committed)
        target.crop(before + len(committed))
        if tokens != reference[:len(tokens)]:
            break
    return tokens


@torch.no_grad()
def fixed_failure_paths(model, ids, example, capacity):
    index = example["first_mismatch_index"]
    sequential, rounds = example["cached_target_tokens"], example["rounds"]
    emitted = 1
    for ri, row in enumerate(rounds):
        if emitted <= index < emitted + row["committed"]:
            offset = index - emitted
            break
        emitted += row["committed"]
    else:
        raise ValueError("No containing round")
    def history(short):
        target = FixedTarget(model, capacity)
        target.prefill(ids)
        anchor = sequential[0]
        for row in rounds[:ri]:
            target.query(ids.new_tensor([[anchor] + row["proposal_tokens"]]))
            target.crop(row["cache_after"])
            anchor = row["committed_tokens"][-1]
        chunk = [anchor] + rounds[ri]["proposal_tokens"]
        logits = target.query(ids.new_tensor([chunk[:offset + 1] if short else chunk]))
        return logits[:, offset].clone()
    full_chunk, short_chunk = history(False), history(True)
    target = FixedTarget(model, capacity)
    target.prefill(ids)
    single = None
    for token in sequential[:index]:
        before = target.length
        single = target.query(ids.new_tensor([[token]]))[:, 0].clone()
        target.crop(before + 1)
    # Canonical route is evaluated on the original prescribed semantic prefix,
    # not on a conveniently changed greedy trajectory.
    return {"index": index, "position_in_block": offset,
        "block_history_full": top(full_chunk), "block_history_short_padded": top(short_chunk),
        "single_token_history_padded": top(single),
        "same_argmax_all_paths": int(full_chunk.argmax()) == int(short_chunk.argmax()) == int(single.argmax()),
        "full_vs_short_max_abs": float((full_chunk.float() - short_chunk.float()).abs().max()),
        "block_vs_single_history_max_abs": float((full_chunk.float() - single.float()).abs().max())}


@torch.no_grad()
def invariants(model, ids, capacity):
    logits, pointers = [], None
    for dummy in (0, 63):
        target = FixedTarget(model, capacity)
        target.prefill(ids)
        before = target.length
        logits.append(target.query(ids.new_tensor([[17]]), dummy=dummy)[:, 0].clone())
        pointers = target.pointers()
        target.crop(before)
        assert target.pointers() == pointers
    dummy_equal = torch.equal(*logits)
    poisoned = []
    for value in (-1000, 1000):
        target = FixedTarget(model, capacity)
        target.prefill(ids)
        start = target.length
        target.query(ids.new_tensor([[17, 18, 19, 20, 21, 22, 23, 24]]))
        target.crop(start + 1)
        for layer in target.cache.layers:
            layer.keys[:, :, target.length:].fill_(value)
            layer.values[:, :, target.length:].fill_(value)
        poisoned.append(target.query(ids.new_tensor([[29]]))[:, 0].clone())
    return {"dummy_suffix_equal": dummy_equal, "stale_suffix_poison_equal": torch.equal(*poisoned),
            "crop_preserves_buffer_addresses": True}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--failure-report", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    source = json.loads(Path(args.failure_report).read_text())
    checkpoint = Path(source["identity"]["args"]["checkpoint"])
    meta = json.loads((checkpoint / "metadata.json").read_text())
    cfg = meta["identity"]["config"]
    records = {r["id"]: r for r in map(json.loads, Path(cfg["records"]).read_text().splitlines())}
    examples = source["examples"]
    capacity = 256
    for ex in examples:
        assert ex["id"] in records
        assert len(records[ex["id"]]["prompt_token_ids"]) + 32 + 8 <= capacity
    if args.dry_run:
        print(json.dumps({"dry_run": True, "examples": len(examples),
            "original_failures": sum(not e["cached_equal"] for e in examples), "capacity": capacity}))
        return
    out = Path(args.output)
    if out.exists(): raise FileExistsError(out)
    torch.set_num_threads(4)
    free, total = torch.cuda.mem_get_info()
    if free < 6 * 1024**3: raise RuntimeError("Less than 6 GiB free; refusing probe")
    torch.cuda.set_per_process_memory_fraction(4 * 1024**3 / total)
    begin = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    target = CachedTarget(model)
    full = FrozenTarget(model, DraftConfig.from_dict(meta["draft_config"]))
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    report = {"scope": "BF16 fixed-shape target-only numerical probe; oracle proposals, not trained draft performance",
        "capacity": capacity, "width": 8, "examples": [],
        "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ("gpu_probe.py", "prototype.py")},
        "failure_report_sha256": hashlib.sha256(Path(args.failure_report).read_bytes()).hexdigest()}
    for number, ex in enumerate(examples):
        ids = torch.tensor([records[ex["id"]]["prompt_token_ids"]], device="cuda", dtype=torch.long)
        result = {"example_index": number, "prompt_tokens": ids.shape[1], "original_failed": not ex["cached_equal"]}
        if not ex["cached_equal"]:
            result["stock_failure_decomposition"] = diagnose_same_prefix(
                target, full, ids, ex["cached_target_tokens"], ex["first_mismatch_index"], ex["rounds"])
        stock = target.greedy(ids, 32, eos)
        with capture_shapes() as shape_events:
            canonical, _ = canonical_sequence(model, ids, capacity, 32, eos)
            speculative = canonical_oracle_block(model, ids, capacity, canonical, 32, eos)
            if not ex["cached_equal"]:
                result["fixed_original_failure_prefix"] = fixed_failure_paths(model, ids, ex, capacity)
            result["invariants"] = invariants(model, ids, capacity)
        decode_shapes = sorted(set(event for event in shape_events if event[0][-2] == 8))
        result.update(stock_tokens=stock, canonical_tokens=canonical, oracle_block_tokens=speculative,
            canonical_vs_stock_equal=canonical == stock, canonical_vs_block_equal=canonical == speculative,
            decode_shapes_strides=decode_shapes, unique_decode_shapes=len(decode_shapes))
        report["examples"].append(result)
        report["elapsed_seconds"] = time.monotonic() - begin
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"example_index": number, "canonical_vs_stock_equal": result["canonical_vs_stock_equal"],
            "canonical_vs_block_equal": result["canonical_vs_block_equal"],
            "fixed_failure_same_argmax": result.get("fixed_original_failure_prefix", {}).get("same_argmax_all_paths"),
            "unique_decode_shapes": len(decode_shapes), "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__": main()
