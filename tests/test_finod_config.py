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
