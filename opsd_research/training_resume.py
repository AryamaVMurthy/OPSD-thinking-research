"""Fail-closed discovery of resumable Hugging Face trainer checkpoints."""

from __future__ import annotations

from pathlib import Path


def latest_valid_checkpoint(output_dir: str | Path) -> Path | None:
    """Return the highest complete checkpoint, ignoring partial directories."""
    root = Path(output_dir)
    if not root.is_dir():
        return None
    valid: list[tuple[int, Path]] = []
    for path in root.glob("checkpoint-*"):
        suffix = path.name.removeprefix("checkpoint-")
        if not suffix.isdigit() or int(suffix) < 1:
            continue
        if not (path / "trainer_state.json").is_file():
            continue
        if not (path / "adapter_model.safetensors").is_file():
            continue
        valid.append((int(suffix), path))
    return max(valid, default=(0, None), key=lambda item: item[0])[1]
