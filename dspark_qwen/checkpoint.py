"""Trainable-only portable weights plus a local, trusted resume state."""
import hashlib
import json
import os
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_checkpoint(root, step, draft, optimizer, generator, metadata):
    root = Path(root)
    destination = root / f"step-{step:06d}"
    temporary = root / f".step-{step:06d}.partial"
    temporary.mkdir(exist_ok=False)
    save_file(draft.trainable_state(), str(temporary / "draft.safetensors"))
    torch.save({"optimizer": optimizer.state_dict(), "anchor_rng": generator.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "step": step}, temporary / "resume.pt")
    (temporary / "metadata.json").write_text(json.dumps({**metadata, "step": step,
        "draft_weights_sha256": sha256(temporary / "draft.safetensors")}, indent=2) + "\n")
    os.replace(temporary, destination)
    (root / "latest.tmp").write_text(destination.name)
    os.replace(root / "latest.tmp", root / "latest")
    return destination


def load_checkpoint(path, draft, expected_identity=None, optimizer=None, generator=None):
    path = Path(path)
    metadata = json.loads((path / "metadata.json").read_text())
    if expected_identity is not None and metadata["identity"] != expected_identity:
        raise ValueError("Checkpoint identity mismatch (source/config/data/model/runtime)")
    if sha256(path / "draft.safetensors") != metadata["draft_weights_sha256"]:
        raise ValueError("Checkpoint weight hash mismatch")
    draft.load_trainable_state(load_file(str(path / "draft.safetensors")))
    if optimizer is not None:
        # weights_only refuses arbitrary pickle globals. Only load our local run files.
        state = torch.load(path / "resume.pt", map_location="cpu", weights_only=True)
        optimizer.load_state_dict(state["optimizer"])
        generator.set_state(state["anchor_rng"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    return metadata
