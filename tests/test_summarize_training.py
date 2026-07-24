from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from opsd_research.summarize_training import summarize


class TrainingSummaryTests(unittest.TestCase):
    def test_summarizes_losses_rollouts_and_checkpoint_integrity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generations = root / "generations"
            generations.mkdir()
            (generations / "generations_step_5.json").write_text(
                json.dumps(
                    {
                        "step": 5,
                        "num_samples": 2,
                        "generations": [
                            {
                                "step": 4,
                                "prompt": "p",
                                "completion": "<think>open",
                            },
                            {
                                "step": 5,
                                "prompt": "q",
                                "completion": "<think>done</think>\\boxed{1}",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            checkpoint = root / "checkpoint-5"
            checkpoint.mkdir()
            (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
            (checkpoint / "trainer_state.json").write_text("{}", encoding="utf-8")
            log = root / "train.log"
            log.write_text(
                "\r  2%| | 4/200 [00:10]\n"
                "{'loss': 0.01, 'grad_norm': 0.2}\n"
                "vLLM generation done - elapsed time: 1.0s, prompts: 1, "
                "total tokens: 1024, avg length: 1024.0\n",
                encoding="utf-8",
            )
            telemetry = root / "gpu.csv"
            telemetry.write_text(
                "2026/01/01 00:00:00.000, 0, NVIDIA A100, 80, 34000, "
                "40960, 200.5, 55\n"
                "2026/01/01 00:00:30.000, 0, NVIDIA A100, 100, 35000, "
                "40960, 250.5, 57\n",
                encoding="utf-8",
            )

            result = summarize(
                root,
                log,
                max_completion_length=1024,
                telemetry=telemetry,
            )

            self.assertEqual(result["latest_logged_step"], 4)
            self.assertTrue(result["loss_finite"])
            self.assertEqual(result["recorded_rollouts"]["count"], 2)
            self.assertEqual(result["recorded_rollouts"]["thinking_closed"], 1)
            self.assertEqual(result["recorded_rollouts"]["boxed_answer"], 1)
            self.assertEqual(result["vllm_rollout_calls"]["at_completion_cap"], 1)
            self.assertTrue(result["checkpoints"][0]["adapter_present"])
            self.assertTrue(result["checkpoints"][0]["trainer_state_present"])
            self.assertTrue(
                result["rollout_dump_integrity"]["covers_latest_checkpoint"]
            )
            gpu = result["gpu_telemetry"][0]
            self.assertEqual(gpu["mean_utilization_percent"], 90)
            self.assertEqual(gpu["max_memory_mib"], 35000)
            self.assertEqual(gpu["max_temperature_c"], 57)


if __name__ == "__main__":
    unittest.main()
