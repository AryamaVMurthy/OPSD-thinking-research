from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ALLOWED_MODELS = {
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
}

MATH_DATASETS = {
    "aime25": {
        "path": "MathArena/aime_2025",
        "split": "train",
        "expected_count": 30,
    },
    "aime26": {
        "path": "MathArena/aime_2026",
        "split": "train",
        "expected_count": 30,
    },
    "hmmt25": {
        "path": "MathArena/hmmt_feb_2025",
        "split": "train",
        "expected_count": 30,
    },
}


class ConfigError(ValueError):
    """Raised when an experiment configuration violates the protocol."""


@dataclass(frozen=True)
class LoadedConfig:
    path: Path
    data: dict[str, Any]


def load_config(path: str | Path) -> LoadedConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ConfigError(f"{config_path}: expected a YAML mapping")
    validate_config(data, source=str(config_path))
    return LoadedConfig(config_path, data)


def _require(data: dict[str, Any], key: str, source: str) -> Any:
    if key not in data:
        raise ConfigError(f"{source}: missing required field {key!r}")
    return data[key]


def _require_thinking(data: dict[str, Any], source: str) -> None:
    if _require(data, "enable_thinking", source) is not True:
        raise ConfigError(f"{source}: enable_thinking must be true")


def _validate_model(data: dict[str, Any], source: str) -> None:
    model = _require(data, "model", source)
    if not isinstance(model, str):
        raise ConfigError(f"{source}: model must be a string")
    if model.endswith("-Base") or "-Base@" in model:
        raise ConfigError(f"{source}: pretrained Qwen3 -Base checkpoints are forbidden")
    if model not in ALLOWED_MODELS:
        raise ConfigError(
            f"{source}: model must be one of {sorted(ALLOWED_MODELS)}, got {model!r}"
        )
    revision = _require(data, "model_revision", source)
    if not isinstance(revision, str) or not revision or revision in {"main", "latest"}:
        raise ConfigError(f"{source}: model_revision must be an immutable commit hash")


def _validate_eval(data: dict[str, Any], source: str) -> None:
    kind = _require(data, "kind", source)
    samples = _require(data, "samples_per_problem", source)
    max_tokens = _require(data, "max_new_tokens", source)
    if kind == "math_eval":
        if samples != 12:
            raise ConfigError(f"{source}: math evaluation requires 12 samples/problem")
        if max_tokens != 38912:
            raise ConfigError(f"{source}: math evaluation requires max_new_tokens=38912")
        if data.get("temperature") != 1.0 or data.get("top_p") != 0.95:
            raise ConfigError(f"{source}: math evaluation requires temperature=1.0, top_p=0.95")
        if data.get("top_k") != -1:
            raise ConfigError(f"{source}: math OPSD evaluation requires top_k=-1")
        dataset = _require(data, "dataset", source)
        if dataset not in MATH_DATASETS:
            raise ConfigError(f"{source}: unknown math dataset alias {dataset!r}")
    elif kind == "lcb_eval":
        if data.get("release_version") != "v6":
            raise ConfigError(f"{source}: LiveCodeBench must use the v6-only slice")
        if samples != 10:
            raise ConfigError(f"{source}: LiveCodeBench requires 10 samples/problem")
        expected = {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "max_new_tokens": 38912,
        }
        for key, value in expected.items():
            if data.get(key) != value:
                raise ConfigError(
                    f"{source}: thinking-enabled LiveCodeBench requires {key}={value}"
                )
    else:
        raise ConfigError(f"{source}: unsupported evaluation kind {kind!r}")


def _validate_train(data: dict[str, Any], source: str) -> None:
    if data.get("kind") != "opsd_train":
        raise ConfigError(f"{source}: unsupported training kind")
    if data.get("student_thinking") is not True:
        raise ConfigError(f"{source}: student_thinking must be true")
    if data.get("teacher_thinking") is not True:
        raise ConfigError(f"{source}: teacher_thinking must be true")
    if data.get("fixed_teacher") is not True:
        raise ConfigError(f"{source}: fixed_teacher must be true")
    expected = {
        "dataset": "jasonrqh/Math-CoT-20k",
        "effective_batch_size": 32,
        "learning_rate": 5e-6,
        "max_completion_length": 1024,
        "max_steps": 200,
        "save_steps": 50,
        "lora_r": 64,
        "lora_alpha": 128,
        "rollouts_per_prompt": 1,
        "temperature": 1.1,
        "top_p": 0.95,
        "top_k": 20,
        "lmbda": 1.0,
        "beta": 0.0,
        "jsd_token_clip": 0.05,
        "seed": 42,
    }
    for key, value in expected.items():
        if data.get(key) != value:
            raise ConfigError(f"{source}: OPSD protocol requires {key}={value!r}")
    computed_batch = (
        int(data.get("per_device_train_batch_size", 0))
        * int(data.get("gradient_accumulation_steps", 0))
        * int(data.get("num_gpus", 0))
    )
    if computed_batch != data["effective_batch_size"]:
        raise ConfigError(
            f"{source}: batch factors produce {computed_batch}, "
            f"not effective_batch_size={data['effective_batch_size']}"
        )
    expected_modules = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
    if set(data.get("lora_target_modules", [])) != expected_modules:
        raise ConfigError(f"{source}: LoRA must target every projection module")


def validate_config(data: dict[str, Any], source: str = "<config>") -> None:
    _validate_model(data, source)
    _require_thinking(data, source)
    kind = _require(data, "kind", source)
    if kind in {"math_eval", "lcb_eval"}:
        _validate_eval(data, source)
    elif kind == "opsd_train":
        _validate_train(data, source)
    else:
        raise ConfigError(f"{source}: unsupported kind {kind!r}")


def discover_configs(root: str | Path) -> list[Path]:
    return sorted(Path(root).glob("reproductions/**/configs/*.yaml"))
