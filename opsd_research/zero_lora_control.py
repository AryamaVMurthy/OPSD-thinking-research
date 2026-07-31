"""Build a same-runtime LoRA placebo with exactly zero adapter scale."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    args = parser.parse_args()
    print(
        json.dumps(
            build_zero_lora_control(args.source, args.output),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
