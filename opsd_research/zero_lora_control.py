"""Build a same-runtime LoRA placebo with exactly zero adapter scale."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_lora_scale(
    source: Path,
    output: Path,
    *,
    scale: float,
) -> dict[str, Any]:
    """Copy a PEFT adapter at a fixed interpolation from the frozen base."""
    source = source.resolve()
    output = output.resolve()
    if not math.isfinite(scale) or not 0.0 <= scale <= 1.0:
        raise ValueError("scale must be finite and in [0, 1]")
    if output.exists():
        raise FileExistsError(f"scaled output already exists: {output}")
    config_path = source / "adapter_config.json"
    weights_path = source / "adapter_model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(
            "source must contain adapter_config.json and "
            "adapter_model.safetensors"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if "lora_alpha" not in config:
        raise ValueError("adapter_config.json has no lora_alpha")
    source_alpha = config["lora_alpha"]
    if not isinstance(source_alpha, (int, float)):
        raise ValueError("lora_alpha must be numeric")
    scaled_alpha: int | float = float(source_alpha) * scale
    if scaled_alpha.is_integer():
        scaled_alpha = int(scaled_alpha)

    output.mkdir(parents=True)
    target_weights = output / weights_path.name
    shutil.copy2(weights_path, target_weights)
    scaled_config = dict(config)
    scaled_config["lora_alpha"] = scaled_alpha
    (output / config_path.name).write_text(
        json.dumps(scaled_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source_sha256 = _sha256(weights_path)
    target_sha256 = _sha256(target_weights)
    if source_sha256 != target_sha256:
        raise RuntimeError("copied adapter weights do not match the source")
    manifest = {
        "schema_version": 1,
        "control": "lora-scale-interpolation-v1",
        "source_adapter": str(source),
        "scale": scale,
        "source_lora_alpha": source_alpha,
        "scaled_lora_alpha": scaled_alpha,
        "source_weights_sha256": source_sha256,
        "weights_sha256": target_sha256,
    }
    (output / "lora-scale-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_zero_lora_control(source: Path, output: Path) -> dict[str, Any]:
    """Copy a PEFT adapter while setting its vLLM contribution scale to zero."""
    source = source.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"control output already exists: {output}")
    config_path = source / "adapter_config.json"
    weights_path = source / "adapter_model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(
            "source must contain adapter_config.json and "
            "adapter_model.safetensors"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if "lora_alpha" not in config:
        raise ValueError("adapter_config.json has no lora_alpha")
    source_alpha = config["lora_alpha"]
    if not isinstance(source_alpha, (int, float)):
        raise ValueError("lora_alpha must be numeric")

    output.mkdir(parents=True)
    target_weights = output / weights_path.name
    shutil.copy2(weights_path, target_weights)
    control_config = dict(config)
    control_config["lora_alpha"] = 0
    (output / config_path.name).write_text(
        json.dumps(control_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source_sha256 = _sha256(weights_path)
    target_sha256 = _sha256(target_weights)
    if source_sha256 != target_sha256:
        raise RuntimeError("copied adapter weights do not match the source")
    manifest = {
        "schema_version": 1,
        "control": "zero-lora-scale-v1",
        "source_adapter": str(source),
        "source_lora_alpha": source_alpha,
        "control_lora_alpha": 0,
        "source_weights_sha256": source_sha256,
        "weights_sha256": target_sha256,
    }
    (output / "zero-control-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scale", type=float, default=0.0)
    args = parser.parse_args()
    builder = (
        build_zero_lora_control
        if args.scale == 0.0
        else lambda source, output: build_lora_scale(
            source,
            output,
            scale=args.scale,
        )
    )
    print(
        json.dumps(
            builder(args.source, args.output),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
