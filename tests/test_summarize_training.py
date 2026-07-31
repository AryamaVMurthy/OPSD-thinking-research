from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from opsd_research.summarize_training import summarize
from opsd_research.summarize_training import _parse_finod_events, _parse_training_log


class TrainingSummaryTests(unittest.TestCase):
    def test_pre_update_rollout_dump_covers_the_next_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generations = root / "generations"
            generations.mkdir()
            (generations / "generations_step_5.json").write_text(
                json.dumps(
                    {
                        "step": 5,
                        "num_samples": 1,
                        "generations": [
                            {
                                "step": 5,
                                "prompt": "problem",
                                "completion": "<think>rollout",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            checkpoint = root / "checkpoint-6"
            checkpoint.mkdir()
            (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
            (checkpoint / "trainer_state.json").write_text(
                "{}", encoding="utf-8"
            )
            log = root / "train.log"
            log.write_text("", encoding="utf-8")

            result = summarize(
                root,
                log,
                max_completion_length=1024,
            )

            integrity = result["rollout_dump_integrity"]
            self.assertEqual(integrity["generation_checkpoint_lag"], 1)
            self.assertTrue(integrity["covers_latest_checkpoint"])

    def test_parses_unrounded_finod_signal_events(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "train.log"
            event = {
                "event": "finod_loss",
                "loss": 0.003,
                "residual_energy": 0.2,
                "target_kl": 0.003,
            }
            log.write_text(
                "ordinary output\n" + json.dumps(event) + "\n",
                encoding="utf-8",
            )

            self.assertEqual(_parse_finod_events(log), [event])

    def test_resumed_log_keeps_last_loss_for_replayed_step(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "train.log"
            log.write_text(
                "\r  25%| | 50/200 [01:00]\n"
                "{'loss': 0.5, 'grad_norm': 0.2}\n"
                "\r  25%| | 50/200 [00:01]\n"
                "{'loss': 0.4, 'grad_norm': 0.1}\n",
                encoding="utf-8",
            )

            losses, _ = _parse_training_log(log)

            self.assertEqual(
                losses,
                [{"step": 50, "loss": 0.4, "grad_norm": 0.1}],
            )

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
            derived_checkpoint = root / "checkpoint-5-scaled-3of8"
            derived_checkpoint.mkdir()
            (derived_checkpoint / "adapter_model.safetensors").write_bytes(
                b"derived adapter"
            )
            log = root / "train.log"
            log.write_text(
                "\r  2%| | 4/200 [00:10]\n"
                "{'loss': 0.01, 'grad_norm': 0.2}\n"
                + json.dumps(
                    {
                        "event": "fisher_consensus_loss",
                        "loss": 0.003,
                        "agreement": 0.75,
                        "mean_direction_energy": 0.08,
                        "consensus_energy": 0.05,
                        "target_forward_kl": 0.003,
                        "target_reverse_kl": 0.0029,
                        "target_kl": 0.0031,
                        "max_observed_target_kl": 0.01,
                        "clipped_target_fraction": 0.2,
                        "collapsed_consensus_fraction": 0.25,
                        "relative_loss_to_anchor": 1.1,
                        "student_anchor_forward_kl": 0.0005,
                        "fisher_alignment_gain": 0.0002,
                        "fisher_alignment_cosine_proxy": 0.04,
                        "entropy_gradient_energy": 0.12,
                        "entropy_alignment_before": -0.03,
                        "entropy_alignment_after": 2.0e-10,
                        "first_order_entropy_change": -2.0e-10,
                        "retained_direction_energy_fraction": 0.65,
                        "target_entropy_change": -0.001,
                        "retraction_mode": "self_information_mixture",
                        "positivity_limited_fraction": 0.3,
                        "mean_minimum_mixture_ratio": 0.05,
                        "mean_absolute_base_cross_entropy_change": 1.0e-8,
                        "mean_absolute_entropy_kl_identity_residual": 2.0e-8,
                    }
                )
                + "\n"
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
            self.assertEqual(len(result["checkpoints"]), 1)
            self.assertTrue(
                result["rollout_dump_integrity"]["covers_latest_checkpoint"]
            )
            fisher = result["fisher_signal"]
            self.assertEqual(fisher["events"], 1)
            self.assertEqual(fisher["post_initial_events"], 1)
            self.assertTrue(fisher["all_finite"])
            self.assertEqual(fisher["positive_loss_events"], 1)
            self.assertEqual(fisher["max_target_kl"], 0.01)
            self.assertEqual(fisher["mean_agreement"], 0.75)
            self.assertEqual(fisher["mean_post_initial_relative_loss_to_anchor"], 1.1)
            self.assertEqual(fisher["mean_post_initial_alignment_gain"], 0.0002)
            self.assertEqual(fisher["mean_post_initial_alignment_cosine_proxy"], 0.04)
            self.assertEqual(fisher["entropy_projection_events"], 1)
            self.assertTrue(fisher["entropy_projection_metrics_complete"])
            self.assertTrue(fisher["entropy_projection_all_finite"])
            self.assertEqual(fisher["mean_entropy_gradient_energy"], 0.12)
            self.assertEqual(
                fisher["mean_absolute_entropy_alignment_before"], 0.03
            )
            self.assertEqual(
                fisher["max_absolute_entropy_alignment_after"], 2.0e-10
            )
            self.assertEqual(
                fisher["mean_retained_direction_energy_fraction"], 0.65
            )
            self.assertEqual(fisher["mean_target_entropy_change"], -0.001)
            self.assertEqual(fisher["self_information_events"], 1)
            self.assertTrue(fisher["self_information_metrics_complete"])
            self.assertTrue(fisher["self_information_all_finite"])
            self.assertEqual(fisher["mean_positivity_limited_fraction"], 0.3)
            self.assertEqual(fisher["mean_minimum_mixture_ratio"], 0.05)
            self.assertEqual(
                fisher["max_mean_absolute_base_cross_entropy_change"],
                1.0e-8,
            )
            self.assertEqual(
                fisher["max_mean_absolute_entropy_kl_identity_residual"],
                2.0e-8,
            )
            gpu = result["gpu_telemetry"][0]
            self.assertEqual(gpu["mean_utilization_percent"], 90)
            self.assertEqual(gpu["max_memory_mib"], 35000)
            self.assertEqual(gpu["max_temperature_c"], 57)


if __name__ == "__main__":
    unittest.main()
