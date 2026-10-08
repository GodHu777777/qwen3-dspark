"""Use audited exact generation tokens, supervising assistant completion + EOS."""
import json
import torch


def load_records(path, split, max_sequence_tokens):
    rows = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if not row["accepted"] or row["split"] != split:
                continue
            ids = row["prompt_token_ids"] + row["output_token_ids"]
            if len(ids) > max_sequence_tokens:
                continue  # Do not silently truncate an accepted answer.
            if not row["template_prefix_matches_generation"] or row["finish_reason"] != "eos":
                raise ValueError("Unaudited completion")
            rows.append(dict(id=row["id"], input_ids=ids, answer_start=len(row["prompt_token_ids"])))
    if not rows:
        raise ValueError(f"No usable records for {split}")
    return rows


def tensors(row, device):
    ids = torch.tensor([row["input_ids"]], device=device, dtype=torch.long)
    answer_mask = torch.arange(ids.shape[1], device=device)[None] >= row["answer_start"]
    return ids, answer_mask


def select_anchors(answer_mask, count, generator):
    # First predicted token a+1 must be an assistant token, including its EOS.
    choices = torch.nonzero(answer_mask[0, 1:], as_tuple=False).flatten().cpu()
    if not choices.numel():
        raise ValueError("No valid anchor")
    picked = choices[torch.randperm(len(choices), generator=generator)[:count]]
    return picked.sort().values.to(answer_mask.device)
