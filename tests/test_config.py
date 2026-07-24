from __future__ import annotations

import copy
import unittest
from pathlib import Path

from opsd_research.config import ConfigError, discover_configs, load_config, validate_config


ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_every_committed_config_is_valid(self):
        configs = discover_configs(ROOT)
        self.assertEqual(len(configs), 10)
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


if __name__ == "__main__":
    unittest.main()
