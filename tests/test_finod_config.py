from pathlib import Path

import pytest

from opsd_research.config import ConfigError, load_config


ROOT = Path(__file__).resolve().parents[1]


def test_finod_screen_config_is_a_valid_eight_gpu_graf_candidate():
    config = load_config(
        ROOT / "reproductions/06_graf_opsd/configs/finod-g1-128.yaml"
    ).data

    assert config["graph_mode"] == "finod_scaffold"
    assert config["num_gpus"] == 8
    assert (
        config["per_device_train_batch_size"]
        * config["gradient_accumulation_steps"]
        * config["num_gpus"]
        == config["effective_batch_size"]
    )
    assert config["finod_positions_per_rollout"] == 32
    assert config["finod_max_target_kl"] == 0.01


def test_finod_representative_config_registers_exact_one_epoch_scale():
    config = load_config(
        ROOT
        / "reproductions/06_graf_opsd/configs/"
        "finod-g1-representative-1024.yaml"
    ).data

    assert config["finod_max_records"] == 1024
    assert config["finod_selection_seed"] == 73
    assert config["finod_guidance_input_protocol"] == "problem-only-v1"
    assert (
        config["finod_answer_leakage_protocol"]
        == "surface-equivalence-and-result-claim-v2"
    )
    assert config["max_completion_length"] == 4096
    assert config["max_steps"] == 32
    assert config["save_steps"] == 32


def test_signed_orthogonal_finod_config_is_registered():
    config = load_config(
        ROOT
        / "reproductions/06_graf_opsd/configs/"
        "finod-g2-orthogonal-representative-1024.yaml"
    ).data

    assert config["finod_projection_mode"] == "signed-orthogonal-v1"
    assert config["finod_max_records"] == 1024
    assert config["max_steps"] == 32


def test_entropy_neutral_finod_config_is_registered():
    config = load_config(
        ROOT
        / "reproductions/06_graf_opsd/configs/"
        "finod-g3-entropy-neutral-representative-1024.yaml"
    ).data

    assert (
        config["finod_projection_mode"]
        == "entropy-neutral-one-sided-v1"
    )
    assert config["finod_max_records"] == 1024
    assert config["max_steps"] == 32


def test_fixed_anchor_style_residual_screen_is_registered():
    config = load_config(
        ROOT
        / "reproductions/06_graf_opsd/configs/"
        "finod-s1-style-fixed-1024.yaml"
    ).data

    assert config["finod_nuisance_view"] == "style"
    assert config["finod_projection_mode"] == "signed-orthogonal-v1"
    assert config["finod_max_records"] == 1024
    assert config["max_steps"] == 5
    assert config["save_steps"] == 5


def test_contest_prefix_screen_is_registered():
    config = load_config(
        ROOT
        / "reproductions/06_graf_opsd/configs/"
        "finod-s2-contest-prefix.yaml"
    ).data

    assert config["finod_data_sources"] == ["amc_aime", "aops_forum"]
    assert config["finod_position_prefix_tokens"] == 1024
    assert config["finod_nuisance_view"] == "style"
    assert config["finod_projection_mode"] == "signed-orthogonal-v1"
    assert config["max_steps"] == 5


def test_answer_free_fisher_consensus_screen_is_registered():
    config = load_config(
        ROOT
        / "reproductions/06_graf_opsd/configs/"
        "fisher-consensus-s1-contest-prefix.yaml"
    ).data

    assert config["graph_mode"] == "fisher_consensus"
    assert config["fisher_data_sources"] == ["amc_aime", "aops_forum"]
    assert config["fisher_guidance_input_protocol"] == (
        "problem-only-independent-v1"
    )
    assert config["fisher_plans_per_problem"] == 3
    assert config["fisher_max_records"] == 192
    assert config["fisher_position_prefix_tokens"] == 1024
    assert config["max_completion_length"] == 1024
    assert config["num_gpus"] == 8
    assert config["max_steps"] == 5


def test_answer_free_fisher_consensus_four_gpu_fallback_preserves_batch():
    config = load_config(
        ROOT
        / "reproductions/06_graf_opsd/configs/"
        "fisher-consensus-s1-contest-prefix-4gpu.yaml"
    ).data

    assert config["graph_mode"] == "fisher_consensus"
    assert config["num_gpus"] == 4
    assert config["gradient_accumulation_steps"] == 8
    assert (
        config["num_gpus"]
        * config["per_device_train_batch_size"]
        * config["gradient_accumulation_steps"]
        == config["effective_batch_size"]
        == 32
    )
    assert config["fisher_max_records"] == 192
    assert config["max_completion_length"] == 1024


def test_fisher_consensus_two_epoch_diagnostic_uses_small_optimizer_steps():
    config = load_config(
        ROOT
        / "reproductions/06_graf_opsd/configs/"
        "fisher-consensus-s2-low-lr-12.yaml"
    ).data

    assert config["graph_mode"] == "fisher_consensus"
    assert config["learning_rate"] == 1e-6
    assert config["max_steps"] == config["save_steps"] == 12
    assert config["fisher_max_records"] == 192
    assert config["max_completion_length"] == 1024
    assert config["num_gpus"] == 8


def test_fisher_consensus_rejects_a_subset_without_equal_domain_quotas(
    tmp_path,
):
    source = (
        ROOT
        / "reproductions/06_graf_opsd/configs/"
        "fisher-consensus-s1-contest-prefix.yaml"
    )
    invalid = tmp_path / "fisher-imbalanced.yaml"
    invalid.write_text(
        source.read_text(encoding="utf-8").replace(
            "fisher_max_records: 192",
            "fisher_max_records: 194",
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ConfigError,
        match="divisible by four AIME domains",
    ):
        load_config(invalid)
