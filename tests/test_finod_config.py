from pathlib import Path

from opsd_research.config import load_config


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
