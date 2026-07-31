import json

import pytest

from opsd_research.zero_lora_control import build_zero_lora_control


def test_build_zero_lora_control_preserves_weights_and_zeroes_scale(
    tmp_path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": 64,
                "lora_alpha": 128,
                "target_modules": ["q_proj"],
            }
        ),
        encoding="utf-8",
    )
    weights = b"synthetic-safetensors"
    (source / "adapter_model.safetensors").write_bytes(weights)
    output = tmp_path / "zero"

    manifest = build_zero_lora_control(source, output)

    zero_config = json.loads(
        (output / "adapter_config.json").read_text(encoding="utf-8")
    )
    source_config = json.loads(
        (source / "adapter_config.json").read_text(encoding="utf-8")
    )
    assert zero_config["lora_alpha"] == 0
    assert source_config["lora_alpha"] == 128
    assert (output / "adapter_model.safetensors").read_bytes() == weights
    assert manifest["source_lora_alpha"] == 128
    assert manifest["control_lora_alpha"] == 0
    assert manifest["weights_sha256"] == manifest["source_weights_sha256"]


def test_build_zero_lora_control_refuses_existing_output(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "adapter_config.json").write_text(
        '{"r": 1, "lora_alpha": 1, "target_modules": ["q_proj"]}',
        encoding="utf-8",
    )
    (source / "adapter_model.safetensors").write_bytes(b"weights")
    output = tmp_path / "zero"
    output.mkdir()

    with pytest.raises(FileExistsError):
        build_zero_lora_control(source, output)
