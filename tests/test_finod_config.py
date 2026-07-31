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


def test_fisher_consensus_high_epsilon_screen_calibrates_tiny_residuals():
    config = load_config(
        ROOT
        / "reproductions/06_graf_opsd/configs/"
        "fisher-consensus-s3-adam-eps-12.yaml"
    ).data

    assert config["graph_mode"] == "fisher_consensus"
    assert config["learning_rate"] == 1e-6
    assert config["adam_epsilon"] == 1e-4
    assert config["max_steps"] == config["save_steps"] == 12
    assert config["fisher_max_records"] == 192


def test_fisher_consensus_proximal_screen_uses_frozen_policy_kl():
    config = load_config(
        ROOT
        / "reproductions/06_graf_opsd/configs/"
        "fisher-consensus-s4-proximal-12.yaml"
    ).data

    assert config["graph_mode"] == "fisher_consensus"
    assert config["adam_epsilon"] == 1e-4
    assert config["fisher_anchor_kl_weight"] == 1.0
    assert config["fisher_max_records"] == 192
    assert config["max_steps"] == 12


def test_fisher_consensus_stronger_target_keeps_hard_kl_cap():
    config = load_config(
        ROOT
        / "reproductions/06_graf_opsd/configs/"
        "fisher-consensus-s5-strong-proximal-6.yaml"
    ).data

    assert config["graph_mode"] == "fisher_consensus"
    assert config["fisher_step_size"] == 1.0
    assert config["fisher_max_target_kl"] == 0.01
    assert config["fisher_anchor_kl_weight"] == 1.0
    assert config["max_steps"] == config["save_steps"] == 6


def test_fisher_edge_boosting_screen_requires_cross_problem_dp_batch():
    config = load_config(
        ROOT
        / "reproductions/06_graf_opsd/configs/"
        "fisher-consensus-s6-edge-boost-6.yaml"
    ).data

    assert config["graph_mode"] == "fisher_consensus"
    assert config["fisher_cross_problem_edge"] is True
    assert config["fisher_edge_threshold"] == 0.0
    assert config["num_gpus"] == 8
    assert config["per_device_train_batch_size"] == 1
    assert config["max_steps"] == 6


def test_fisher_plan_barycenter_screen_is_registered_without_edge_gate():
    config = load_config(
        ROOT
        / "reproductions/06_graf_opsd/configs/"
        "fisher-consensus-s7-plan-barycenter-6.yaml"
    ).data

    assert config["fisher_direction_mode"] == "positive_plan_barycenter"
    assert config["fisher_cross_problem_edge"] is False
    assert config["fisher_step_size"] == 1.0
    assert config["fisher_anchor_kl_weight"] == 1.0
    assert config["max_steps"] == 6


def test_fisher_plan_barycenter_strong_proximal_matches_drift_prediction():
    config = load_config(
        ROOT
        / "reproductions/06_graf_opsd/configs/"
        "fisher-consensus-s8-barycenter-prox4-6.yaml"
    ).data

    assert config["fisher_direction_mode"] == "positive_plan_barycenter"
    assert config["fisher_anchor_kl_weight"] == 4.0
    assert config["fisher_step_size"] == 1.0
    assert config["fisher_cross_problem_edge"] is False
    assert config["max_steps"] == 6


def test_promoted_barycenter_scaleup_is_exactly_domain_balanced():
    config = load_config(
        ROOT
        / "reproductions/06_graf_opsd/configs/"
        "fisher-consensus-s9-barycenter-balanced-512.yaml"
    ).data

    assert config["fisher_direction_mode"] == "positive_plan_barycenter"
    assert config["fisher_max_records"] == 512
    assert config["fisher_max_records"] // 4 == 128
    assert config["max_steps"] == config["save_steps"] == 16
    assert config["fisher_anchor_kl_weight"] == 1.0


def test_barycenter_prefix_screen_changes_only_supervised_horizon():
    baseline = load_config(
        ROOT
        / "reproductions/06_graf_opsd/configs/"
        "fisher-consensus-s7-plan-barycenter-6.yaml"
    ).data
    candidate = load_config(
        ROOT
        / "reproductions/06_graf_opsd/configs/"
        "fisher-consensus-s10-barycenter-prefix512-6.yaml"
    ).data

    assert candidate["fisher_position_prefix_tokens"] == 512
    ignored = {"variant", "fisher_position_prefix_tokens"}
    assert {
        key: value for key, value in candidate.items() if key not in ignored
    } == {
        key: value for key, value in baseline.items() if key not in ignored
    }


def test_barycenter_global_batch_screen_changes_only_update_grouping():
    baseline = load_config(
        ROOT
        / "reproductions/06_graf_opsd/configs/"
        "fisher-consensus-s7-plan-barycenter-6.yaml"
    ).data
    candidate = load_config(
        ROOT
        / "reproductions/06_graf_opsd/configs/"
        "fisher-consensus-s11-barycenter-global-batch-1.yaml"
    ).data

    assert candidate["effective_batch_size"] == 192
    assert candidate["gradient_accumulation_steps"] == 24
    assert candidate["max_steps"] == candidate["save_steps"] == 1
    assert candidate["fisher_max_records"] == 192
    ignored = {
        "variant",
        "effective_batch_size",
        "gradient_accumulation_steps",
        "max_steps",
        "save_steps",
    }
    assert {
        key: value for key, value in candidate.items() if key not in ignored
    } == {
        key: value for key, value in baseline.items() if key not in ignored
    }


def test_barycenter_prefix512_global_batch_combines_only_screened_axes():
    full_prefix_global = load_config(
        ROOT
        / "reproductions/06_graf_opsd/configs/"
        "fisher-consensus-s11-barycenter-global-batch-1.yaml"
    ).data
    prefix512_sequential = load_config(
        ROOT
        / "reproductions/06_graf_opsd/configs/"
        "fisher-consensus-s10-barycenter-prefix512-6.yaml"
    ).data
    candidate = load_config(
        ROOT
        / "reproductions/06_graf_opsd/configs/"
        "fisher-consensus-s12-barycenter-prefix512-global-batch-1.yaml"
    ).data

    assert candidate["fisher_position_prefix_tokens"] == 512
    ignored_from_global = {"variant", "fisher_position_prefix_tokens"}
    assert {
        key: value for key, value in candidate.items() if key not in ignored_from_global
    } == {
        key: value
        for key, value in full_prefix_global.items()
        if key not in ignored_from_global
    }

    grouping_fields = {
        "variant",
        "effective_batch_size",
        "gradient_accumulation_steps",
        "max_steps",
        "save_steps",
    }
    assert {
        key: value for key, value in candidate.items() if key not in grouping_fields
    } == {
        key: value
        for key, value in prefix512_sequential.items()
        if key not in grouping_fields
    }


def test_entropy_neutral_barycenter_changes_only_fisher_projection():
    baseline = load_config(
        ROOT
        / "reproductions/06_graf_opsd/configs/"
        "fisher-consensus-s10-barycenter-prefix512-6.yaml"
    ).data
    candidate = load_config(
        ROOT
        / "reproductions/06_graf_opsd/configs/"
        "fisher-consensus-s13-entropy-neutral-prefix512-6.yaml"
    ).data

    assert (
        candidate["fisher_direction_mode"]
        == "entropy_neutral_plan_barycenter"
    )
    ignored = {"variant", "fisher_direction_mode"}
    assert {
        key: value for key, value in candidate.items() if key not in ignored
    } == {
        key: value for key, value in baseline.items() if key not in ignored
    }


def test_self_information_mixture_changes_only_fisher_retraction():
    baseline = load_config(
        ROOT
        / "reproductions/06_graf_opsd/configs/"
        "fisher-consensus-s13-entropy-neutral-prefix512-6.yaml"
    ).data
    candidate = load_config(
        ROOT
        / "reproductions/06_graf_opsd/configs/"
        "fisher-consensus-s14-self-information-mixture-prefix512-6.yaml"
    ).data

    assert candidate["fisher_retraction_mode"] == "self_information_mixture"
    ignored = {"variant", "fisher_retraction_mode"}
    assert {
        key: value for key, value in candidate.items() if key not in ignored
    } == {
        key: value for key, value in baseline.items() if key not in ignored
    }


def test_self_information_exponential_changes_only_fisher_retraction():
    baseline = load_config(
        ROOT
        / "reproductions/06_graf_opsd/configs/"
        "fisher-consensus-s13-entropy-neutral-prefix512-6.yaml"
    ).data
    candidate = load_config(
        ROOT
        / "reproductions/06_graf_opsd/configs/"
        "fisher-consensus-s15-self-information-exponential-prefix512-6.yaml"
    ).data

    assert candidate["fisher_retraction_mode"] == (
        "self_information_exponential"
    )
    ignored = {"variant", "fisher_retraction_mode"}
    assert {
        key: value for key, value in candidate.items() if key not in ignored
    } == {
        key: value for key, value in baseline.items() if key not in ignored
    }


def test_decisive_core_screen_changes_only_plan_span():
    baseline = load_config(
        ROOT
        / "reproductions/06_graf_opsd/configs/"
        "fisher-consensus-s10-barycenter-prefix512-6.yaml"
    ).data
    candidate = load_config(
        ROOT
        / "reproductions/06_graf_opsd/configs/"
        "fisher-consensus-s16-decisive-core-prefix512-6.yaml"
    ).data

    assert candidate["fisher_plan_core_sentences"] == 2
    ignored = {"variant", "fisher_plan_core_sentences"}
    assert {
        key: value for key, value in candidate.items() if key not in ignored
    } == {
        key: value for key, value in baseline.items() if key not in ignored
    }
