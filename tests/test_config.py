from __future__ import annotations

import copy
import unittest
from pathlib import Path

from opsd_research.config import ConfigError, discover_configs, load_config, validate_config


ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_every_committed_config_is_valid(self):
        configs = discover_configs(ROOT)
        # New pre-registered experimental candidates legitimately add configs;
        # the invariant is that discovery finds the established baseline set
        # and every discovered config validates, not a frozen file count.
        self.assertGreaterEqual(len(configs), 15)
        for path in configs:
            with self.subTest(path=path):
                load_config(path)

    def test_base_model_is_rejected(self):
        data = load_config(
            ROOT
            / "reproductions/01_qwen3_thinking_math/configs/qwen3-1p7b-aime25.yaml"
        ).data
        changed = copy.deepcopy(data)
        changed["model"] = "Qwen/Qwen3-1.7B-Base"
        with self.assertRaisesRegex(ConfigError, "Base"):
            validate_config(changed)

    def test_thinking_off_is_rejected(self):
        data = load_config(
            ROOT / "reproductions/03_opsd_thinking_1p7b/configs/train.yaml"
        ).data
        changed = copy.deepcopy(data)
        changed["student_thinking"] = False
        with self.assertRaisesRegex(ConfigError, "student_thinking"):
            validate_config(changed)

    def test_bad_effective_batch_is_rejected(self):
        data = load_config(
            ROOT / "reproductions/04_opsd_thinking_4b/configs/train.yaml"
        ).data
        changed = copy.deepcopy(data)
        changed["gradient_accumulation_steps"] = 2
        with self.assertRaisesRegex(ConfigError, "batch factors"):
            validate_config(changed)

    def test_bad_exact_jsd_chunk_size_is_rejected(self):
        data = load_config(
            ROOT / "reproductions/04_opsd_thinking_4b/configs/train.yaml"
        ).data
        changed = copy.deepcopy(data)
        changed["exact_jsd_vocab_chunk_size"] = 0
        with self.assertRaisesRegex(ConfigError, "positive integer"):
            validate_config(changed)

    def test_4b_training_requires_tail_logits(self):
        data = load_config(
            ROOT / "reproductions/04_opsd_thinking_4b/configs/train.yaml"
        ).data
        changed = copy.deepcopy(data)
        del changed["tail_logits_only"]
        with self.assertRaisesRegex(ConfigError, "tail_logits_only"):
            validate_config(changed)

    def test_graf_candidate_allows_longer_rollouts_but_not_protocol_drift(self):
        data = load_config(
            ROOT / "reproductions/06_graf_opsd/configs/c0-long-rollout.yaml"
        ).data
        validate_config(data)
        changed = copy.deepcopy(data)
        changed["max_completion_length"] = 1536
        with self.assertRaisesRegex(ConfigError, "max_completion_length"):
            validate_config(changed)

    def test_viability_routed_candidate_requires_a_real_branch_loss(self):
        data = load_config(
            ROOT / "reproductions/06_graf_opsd/configs/g2-viability-routed.yaml"
        ).data
        changed = copy.deepcopy(data)
        changed["branch_loss_weight"] = 0.0
        with self.assertRaisesRegex(ConfigError, "branch_loss_weight"):
            validate_config(changed)

    def test_fork_threshold_requires_routed_mode_and_probability_margin(self):
        data = load_config(
            ROOT / "reproductions/06_graf_opsd/configs/g2-viability-routed.yaml"
        ).data
        changed = copy.deepcopy(data)
        changed["fork_threshold"] = 1.1
        with self.assertRaisesRegex(ConfigError, "fork_threshold"):
            validate_config(changed)
        changed = copy.deepcopy(data)
        changed["graph_mode"] = "disabled"
        changed["branch_loss_weight"] = 0.0
        changed["fork_threshold"] = 0.15
        with self.assertRaisesRegex(ConfigError, "branch settings"):
            validate_config(changed)

    def test_fork_information_threshold_requires_routed_mode_and_valid_range(self):
        data = load_config(
            ROOT / "reproductions/06_graf_opsd/configs/g2-viability-routed.yaml"
        ).data
        changed = copy.deepcopy(data)
        changed["fork_information_threshold"] = 1.1
        with self.assertRaisesRegex(ConfigError, "fork_information_threshold"):
            validate_config(changed)
        changed = copy.deepcopy(data)
        changed["graph_mode"] = "disabled"
        changed["branch_loss_weight"] = 0.0
        changed["fork_information_threshold"] = 0.05
        with self.assertRaisesRegex(ConfigError, "branch settings"):
            validate_config(changed)


if __name__ == "__main__":
    unittest.main()
