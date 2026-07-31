from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ALLOWED_MODELS = {
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
}

MATH_DATASETS = {
    "aime24": {
        "split": "train",
        "expected_count": 30,
        "components": (
            ("I", "MathArena/aime_2024_I"),
            ("II", "MathArena/aime_2024_II"),
        ),
    },
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
        protocol = data.get("evaluation_protocol", "official")
        if protocol == "official":
            expected_samples, expected_tokens = 12, 38912
        elif protocol == "development":
            expected_samples, expected_tokens = 4, 4096
            if data.get("max_model_len") != 6144:
                raise ConfigError(
                    f"{source}: development math evaluation requires max_model_len=6144"
                )
        elif protocol == "development_full_context":
            expected_samples, expected_tokens = 4, 38912
            if data.get("max_model_len") != 40960:
                raise ConfigError(
                    f"{source}: full-context development evaluation requires "
                    "max_model_len=40960"
                )
        elif protocol == "six_sample_32k":
            expected_samples, expected_tokens = 6, 32768
            if data.get("max_model_len") != 40960:
                raise ConfigError(
                    f"{source}: six-sample 32k math evaluation requires "
                    "max_model_len=40960"
                )
        else:
            raise ConfigError(
                f"{source}: evaluation_protocol must be official, development, "
                "development_full_context, or six_sample_32k"
            )
        if samples != expected_samples:
            raise ConfigError(
                f"{source}: {protocol} math evaluation requires "
                f"{expected_samples} samples/problem"
            )
        if max_tokens != expected_tokens:
            raise ConfigError(
                f"{source}: {protocol} math evaluation requires "
                f"max_new_tokens={expected_tokens}"
            )
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
    chunk_size = data.get("exact_jsd_vocab_chunk_size")
    if chunk_size is not None and (
        not isinstance(chunk_size, int)
        or isinstance(chunk_size, bool)
        or chunk_size <= 0
    ):
        raise ConfigError(
            f"{source}: exact_jsd_vocab_chunk_size must be a positive integer"
        )
    tail_logits_only = data.get("tail_logits_only")
    if tail_logits_only not in (None, True):
        raise ConfigError(f"{source}: tail_logits_only must be true when set")
    if data["model"] == "Qwen/Qwen3-4B" and tail_logits_only is not True:
        raise ConfigError(
            f"{source}: Qwen3-4B training requires tail_logits_only=true"
        )


def _validate_graf_train(data: dict[str, Any], source: str) -> None:
    """Validate an experimental GRAF candidate without weakening OPSD checks."""
    if data.get("kind") != "graf_train":
        raise ConfigError(f"{source}: unsupported GRAF training kind")
    if data.get("model") != "Qwen/Qwen3-4B":
        raise ConfigError(f"{source}: GRAF autoresearch is Qwen3-4B only")
    if not isinstance(data.get("variant"), str) or not data["variant"]:
        raise ConfigError(f"{source}: GRAF candidate requires a nonempty variant")
    for key in ("student_thinking", "teacher_thinking", "fixed_teacher"):
        if data.get(key) is not True:
            raise ConfigError(f"{source}: {key} must be true")
    expected = {
        "dataset": "jasonrqh/Math-CoT-20k",
        "effective_batch_size": 32,
        "lora_r": 64,
        "lora_alpha": 128,
        "rollouts_per_prompt": 1,
        "temperature": 1.1,
        "top_p": 0.95,
        "top_k": 20,
        "lmbda": 1.0,
        "beta": 0.0,
        "tail_logits_only": True,
    }
    for key, value in expected.items():
        if data.get(key) != value:
            raise ConfigError(f"{source}: GRAF protocol requires {key}={value!r}")
    seed = data.get("seed")
    if seed not in {42, 43}:
        raise ConfigError(f"{source}: GRAF seed must be one of [42, 43]")
    token_divergence = data.get("token_divergence")
    if token_divergence is None:
        if data.get("jsd_token_clip") != 0.05:
            raise ConfigError(
                f"{source}: legacy GRAF protocol requires jsd_token_clip=0.05"
            )
    else:
        if token_divergence not in {"forward_kl", "reverse_kl", "js"}:
            raise ConfigError(
                f"{source}: token_divergence must be forward_kl, reverse_kl, or js"
            )
        if data.get("jsd_token_clip") is not None:
            raise ConfigError(
                f"{source}: canonical token divergences must be unclipped"
            )
    chunk_size = data.get("exact_jsd_vocab_chunk_size")
    if (
        not isinstance(chunk_size, int)
        or isinstance(chunk_size, bool)
        or chunk_size <= 0
    ):
        raise ConfigError(
            f"{source}: exact_jsd_vocab_chunk_size must be a positive integer"
        )
    if data.get("max_completion_length") not in {1024, 2048, 4096}:
        raise ConfigError(
            f"{source}: max_completion_length must be 1024, 2048, or 4096"
        )
    allowed_steps = {5, 12, 25, 32, 50, 200}
    if data.get("graph_mode") == "fisher_consensus":
        # Six effective-batch updates are one exact pass over the registered
        # 192-example equal-domain Fisher screen.
        allowed_steps.add(6)
    if data.get("max_steps") not in allowed_steps:
        raise ConfigError(
            f"{source}: max_steps is not registered for this GRAF mode"
        )
    save_steps = data.get("save_steps")
    checkpointed_confirmation = (
        data.get("max_steps") == 50 and save_steps == 25
    )
    checkpointed_full_run = (
        data.get("max_steps") == 200 and save_steps == 50
    )
    if (
        save_steps != data.get("max_steps")
        and not checkpointed_confirmation
        and not checkpointed_full_run
    ):
        raise ConfigError(
            f"{source}: pilots save only at the final step; "
            "50-step confirmations may save every 25 steps and "
            "200-step confirmations may save every 50 steps"
        )
    max_sequence_length = data.get("max_sequence_length", 28672)
    if (
        not isinstance(max_sequence_length, int)
        or isinstance(max_sequence_length, bool)
        or max_sequence_length < data["max_completion_length"]
    ):
        raise ConfigError(
            f"{source}: max_sequence_length must cover max_completion_length"
        )
    vllm_memory_fraction = data.get("vllm_gpu_memory_utilization")
    if vllm_memory_fraction is not None and (
        not isinstance(vllm_memory_fraction, (int, float))
        or isinstance(vllm_memory_fraction, bool)
        or not 0.0 < float(vllm_memory_fraction) <= 1.0
    ):
        raise ConfigError(
            f"{source}: vllm_gpu_memory_utilization must be in (0, 1]"
        )
    heldout_fraction = data.get("heldout_diagnostic_fraction", 0.0)
    if (
        not isinstance(heldout_fraction, (int, float))
        or isinstance(heldout_fraction, bool)
        or not 0.0 <= float(heldout_fraction) < 0.5
    ):
        raise ConfigError(
            f"{source}: heldout_diagnostic_fraction must be in [0, 0.5)"
        )
    computed_batch = (
        int(data.get("per_device_train_batch_size", 0))
        * int(data.get("gradient_accumulation_steps", 0))
        * int(data.get("num_gpus", 0))
    )
    if computed_batch != 32:
        raise ConfigError(f"{source}: GRAF batch factors must produce 32")
    if set(data.get("lora_target_modules", [])) != {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    }:
        raise ConfigError(f"{source}: GRAF LoRA target modules are incomplete")
    if data.get("graph_mode") not in {
        "disabled",
        "scaffold_graph",
        "fork_mask",
        "viability_routed",
        "fluid_viability_routed",
        "fluid_context_dossier",
        "context_dossier",
        "finod_scaffold",
        "fisher_consensus",
    }:
        raise ConfigError(f"{source}: unsupported graph_mode")
    allowed_learning_rates = (
        {1e-6, 5e-6}
        if data["graph_mode"] == "fisher_consensus"
        else {5e-6}
    )
    if data.get("learning_rate") not in allowed_learning_rates:
        raise ConfigError(
            f"{source}: {data['graph_mode']} learning_rate must be one of "
            f"{sorted(allowed_learning_rates)}"
        )
    for key in ("branch_loss_weight", "entropy_floor_weight"):
        value = data.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            raise ConfigError(f"{source}: {key} must be a nonnegative number")
    fork_threshold = data.get("fork_threshold", 0.0)
    if (
        not isinstance(fork_threshold, (int, float))
        or isinstance(fork_threshold, bool)
        or not 0.0 <= float(fork_threshold) <= 1.0
    ):
        raise ConfigError(f"{source}: fork_threshold must be a probability margin in [0, 1]")
    fork_information_threshold = data.get("fork_information_threshold", 0.0)
    if (
        not isinstance(fork_information_threshold, (int, float))
        or isinstance(fork_information_threshold, bool)
        or not 0.0 <= float(fork_information_threshold) <= 1.0
    ):
        raise ConfigError(
            f"{source}: fork_information_threshold must be a target-KL threshold in [0, 1]"
        )
    fork_information_quantile = data.get("fork_information_quantile", 0.0)
    if (
        not isinstance(fork_information_quantile, (int, float))
        or isinstance(fork_information_quantile, bool)
        or not 0.0 <= float(fork_information_quantile) <= 1.0
    ):
        raise ConfigError(
            f"{source}: fork_information_quantile must be a cache quantile in [0, 1]"
        )
    if float(fork_information_threshold) and float(fork_information_quantile):
        raise ConfigError(
            f"{source}: choose either fork_information_threshold or fork_information_quantile"
        )
    information_weighted_routing = data.get("information_weighted_routing", False)
    if not isinstance(information_weighted_routing, bool):
        raise ConfigError(f"{source}: information_weighted_routing must be boolean")
    viability_beta_prior = data.get("viability_beta_prior", 0.0)
    if (
        not isinstance(viability_beta_prior, (int, float))
        or isinstance(viability_beta_prior, bool)
        or not math.isfinite(float(viability_beta_prior))
        or float(viability_beta_prior) < 0.0
    ):
        raise ConfigError(f"{source}: viability_beta_prior must be a finite nonnegative number")
    recovery_conditioned_routing = data.get("recovery_conditioned_routing", False)
    if not isinstance(recovery_conditioned_routing, bool):
        raise ConfigError(f"{source}: recovery_conditioned_routing must be boolean")
    teacher_graph_critique = data.get("teacher_graph_critique", False)
    if not isinstance(teacher_graph_critique, bool):
        raise ConfigError(f"{source}: teacher_graph_critique must be boolean")
    if information_weighted_routing and (
        float(fork_information_threshold) or float(fork_information_quantile)
    ):
        raise ConfigError(
            f"{source}: information_weighted_routing cannot be combined with an information cutoff"
        )
    if data.get("graph_mode") in {
        "viability_routed",
        "fluid_viability_routed",
    }:
        if float(data["branch_loss_weight"]) <= 0:
            raise ConfigError(
                f"{source}: routed modes require branch_loss_weight > 0"
            )
        if float(data["entropy_floor_weight"]) > 1:
            raise ConfigError(f"{source}: entropy_floor_weight is an entropy fraction in [0, 1]")
    elif (
        float(data["branch_loss_weight"]) != 0
        or float(data["entropy_floor_weight"]) != 0
        or float(fork_threshold) != 0
        or float(fork_information_threshold) != 0
        or float(fork_information_quantile) != 0
        or information_weighted_routing
        or float(viability_beta_prior) != 0
        or recovery_conditioned_routing
        or teacher_graph_critique
    ):
        raise ConfigError(f"{source}: branch settings require graph_mode=viability_routed")
    if data.get("graph_mode") == "finod_scaffold":
        max_records = data.get("finod_max_records")
        selection_seed = data.get("finod_selection_seed")
        guidance_protocol = data.get("finod_guidance_input_protocol")
        leakage_protocol = data.get("finod_answer_leakage_protocol")
        projection_mode = data.get(
            "finod_projection_mode", "one-sided-positive-v1"
        )
        nuisance_view = data.get("finod_nuisance_view", "answer")
        data_sources = data.get("finod_data_sources")
        position_prefix_tokens = data.get(
            "finod_position_prefix_tokens"
        )
        if max_records is not None:
            if (
                not isinstance(max_records, int)
                or isinstance(max_records, bool)
                or not 32 <= max_records <= 4096
            ):
                raise ConfigError(
                    f"{source}: finod_max_records must be in [32, 4096]"
                )
            if (
                not isinstance(selection_seed, int)
                or isinstance(selection_seed, bool)
                or selection_seed < 0
            ):
                raise ConfigError(
                    f"{source}: representative FiNOD selection requires a "
                    "nonnegative integer finod_selection_seed"
                )
        elif selection_seed is not None:
            raise ConfigError(
                f"{source}: finod_selection_seed requires finod_max_records"
            )
        if guidance_protocol is not None and (
            not isinstance(guidance_protocol, str) or not guidance_protocol
        ):
            raise ConfigError(
                f"{source}: finod_guidance_input_protocol must be a nonempty string"
            )
        if leakage_protocol is not None and (
            not isinstance(leakage_protocol, str) or not leakage_protocol
        ):
            raise ConfigError(
                f"{source}: finod_answer_leakage_protocol must be a nonempty string"
            )
        if projection_mode not in {
            "one-sided-positive-v1",
            "signed-orthogonal-v1",
            "entropy-neutral-one-sided-v1",
        }:
            raise ConfigError(
                f"{source}: unsupported finod_projection_mode "
                f"{projection_mode!r}"
            )
        if nuisance_view not in {"answer", "style"}:
            raise ConfigError(
                f"{source}: unsupported finod_nuisance_view "
                f"{nuisance_view!r}"
            )
        if data_sources is not None and (
            not isinstance(data_sources, list)
            or not data_sources
            or any(
                not isinstance(item, str) or not item
                for item in data_sources
            )
            or len(set(data_sources)) != len(data_sources)
        ):
            raise ConfigError(
                f"{source}: finod_data_sources must be a nonempty list "
                "of unique strings"
            )
        positions = data.get("finod_positions_per_rollout")
        if (
            not isinstance(positions, int)
            or isinstance(positions, bool)
            or not 1 <= positions <= 256
        ):
            raise ConfigError(
                f"{source}: finod_positions_per_rollout must be in [1, 256]"
            )
        if position_prefix_tokens is not None and (
            not isinstance(position_prefix_tokens, int)
            or isinstance(position_prefix_tokens, bool)
            or not 1
            <= position_prefix_tokens
            <= int(data["max_completion_length"])
        ):
            raise ConfigError(
                f"{source}: finod_position_prefix_tokens must be in "
                "[1, max_completion_length]"
            )
        bounded_positive = {
            "finod_step_size": 2.0,
            "finod_max_target_kl": 0.1,
        }
        for key, upper in bounded_positive.items():
            value = data.get(key)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or not 0.0 < float(value) <= upper
            ):
                raise ConfigError(
                    f"{source}: {key} must be finite in (0, {upper}]"
                )
        for key in (
            "finod_nuisance_strength_threshold",
            "finod_residual_energy_threshold",
        ):
            value = data.get(key)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ConfigError(
                    f"{source}: {key} must be finite and nonnegative"
                )
    if data.get("graph_mode") == "fisher_consensus":
        max_records = data.get("fisher_max_records")
        selection_seed = data.get("fisher_selection_seed")
        if (
            not isinstance(max_records, int)
            or isinstance(max_records, bool)
            or not 32 <= max_records <= 4096
        ):
            raise ConfigError(
                f"{source}: fisher_max_records must be in [32, 4096]"
            )
        if max_records % 4:
            raise ConfigError(
                f"{source}: fisher_max_records must be divisible by four "
                "AIME domains"
            )
        adam_epsilon = data.get("adam_epsilon", 1e-8)
        if (
            not isinstance(adam_epsilon, (int, float))
            or isinstance(adam_epsilon, bool)
            or not math.isfinite(float(adam_epsilon))
            or not 1e-12 <= float(adam_epsilon) <= 1e-2
        ):
            raise ConfigError(
                f"{source}: adam_epsilon must be finite in [1e-12, 1e-2]"
            )
        anchor_weight = data.get("fisher_anchor_kl_weight", 0.0)
        if (
            not isinstance(anchor_weight, (int, float))
            or isinstance(anchor_weight, bool)
            or not math.isfinite(float(anchor_weight))
            or not 0.0 <= float(anchor_weight) <= 100.0
        ):
            raise ConfigError(
                f"{source}: fisher_anchor_kl_weight must be finite in "
                "[0, 100]"
            )
        if (
            not isinstance(selection_seed, int)
            or isinstance(selection_seed, bool)
            or selection_seed < 0
        ):
            raise ConfigError(
                f"{source}: fisher_selection_seed must be nonnegative"
            )
        if (
            data.get("fisher_guidance_input_protocol")
            != "problem-only-independent-v1"
        ):
            raise ConfigError(
                f"{source}: Fisher guidance must use the registered "
                "problem-only protocol"
            )
        data_sources = data.get("fisher_data_sources")
        if (
            not isinstance(data_sources, list)
            or not data_sources
            or any(
                not isinstance(item, str) or not item
                for item in data_sources
            )
            or len(set(data_sources)) != len(data_sources)
        ):
            raise ConfigError(
                f"{source}: fisher_data_sources must be unique strings"
            )
        pair_count = data.get("fisher_plans_per_problem")
        if (
            not isinstance(pair_count, int)
            or isinstance(pair_count, bool)
            or not 2 <= pair_count <= 5
        ):
            raise ConfigError(
                f"{source}: fisher_plans_per_problem must be in [2, 5]"
            )
        positions = data.get("fisher_positions_per_rollout")
        if (
            not isinstance(positions, int)
            or isinstance(positions, bool)
            or not 1 <= positions <= 256
        ):
            raise ConfigError(
                f"{source}: fisher_positions_per_rollout must be in [1, 256]"
            )
        prefix_tokens = data.get("fisher_position_prefix_tokens")
        if (
            not isinstance(prefix_tokens, int)
            or isinstance(prefix_tokens, bool)
            or not 1
            <= prefix_tokens
            <= int(data["max_completion_length"])
        ):
            raise ConfigError(
                f"{source}: fisher_position_prefix_tokens must be in "
                "[1, max_completion_length]"
            )
        bounded_positive = {
            "fisher_step_size": 2.0,
            "fisher_max_target_kl": 0.1,
        }
        for key, upper in bounded_positive.items():
            value = data.get(key)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or not 0.0 < float(value) <= upper
            ):
                raise ConfigError(
                    f"{source}: {key} must be finite in (0, {upper}]"
                )
        energy_threshold = data.get(
            "fisher_consensus_energy_threshold"
        )
        if (
            not isinstance(energy_threshold, (int, float))
            or isinstance(energy_threshold, bool)
            or not math.isfinite(float(energy_threshold))
            or float(energy_threshold) < 0.0
        ):
            raise ConfigError(
                f"{source}: fisher_consensus_energy_threshold must be "
                "finite and nonnegative"
            )


def _validate_graf_autoresearch(data: dict[str, Any], source: str) -> None:
    if data.get("kind") != "graf_autoresearch":
        raise ConfigError(f"{source}: unsupported autoresearch kind")
    if data.get("model") != "Qwen/Qwen3-4B":
        raise ConfigError(f"{source}: autoresearch is Qwen3-4B only")
    if data.get("training_dataset") != "jasonrqh/Math-CoT-20k":
        raise ConfigError(f"{source}: autoresearch training dataset is pinned")
    if data.get("development_benchmark") != "aime24":
        raise ConfigError(f"{source}: AIME24 is the fixed development benchmark")
    if data.get("locked_benchmarks") != ["aime25", "aime26"]:
        raise ConfigError(f"{source}: locked benchmarks must be AIME25 then AIME26")
    if data.get("official_samples_per_problem") != 12:
        raise ConfigError(f"{source}: official evaluations require 12 samples/problem")
    if not isinstance(data.get("promotion_min_avg_at_12_delta"), (int, float)):
        raise ConfigError(f"{source}: promotion threshold must be numeric")
    if data["promotion_min_avg_at_12_delta"] < 0.0:
        raise ConfigError(f"{source}: promotion threshold must be nonnegative")
    if not isinstance(data.get("promotion_requires_positive_ci_lower"), bool):
        raise ConfigError(f"{source}: confidence-bound setting must be boolean")
    if not isinstance(data.get("max_candidates"), int) or not 1 <= data["max_candidates"] <= 24:
        raise ConfigError(f"{source}: max_candidates must be in [1, 24]")
    expected_mutations = {
        "max_completion_length", "heldout_diagnostic_fraction", "graph_mode", "full_graph_method", "teacher_graph_critique", "fork_threshold", "fork_information_threshold", "fork_information_quantile", "information_weighted_routing", "graph_budget", "token_divergence",
        "viability_temperature", "viability_beta_prior", "recovery_conditioned_routing", "branch_loss_weight", "entropy_floor_weight",
    }
    if not set(data.get("allowed_mutations", [])).issubset(expected_mutations):
        raise ConfigError(f"{source}: autoresearch includes a forbidden mutation")


def validate_config(data: dict[str, Any], source: str = "<config>") -> None:
    _validate_model(data, source)
    _require_thinking(data, source)
    kind = _require(data, "kind", source)
    if kind in {"math_eval", "lcb_eval"}:
        _validate_eval(data, source)
    elif kind == "opsd_train":
        _validate_train(data, source)
    elif kind == "graf_train":
        _validate_graf_train(data, source)
    elif kind == "graf_autoresearch":
        _validate_graf_autoresearch(data, source)
    else:
        raise ConfigError(f"{source}: unsupported kind {kind!r}")


def discover_configs(root: str | Path) -> list[Path]:
    return sorted(Path(root).glob("reproductions/**/configs/*.yaml"))
