from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from opsd_research.run_report import write_evaluation_report, write_training_report


class RunReportTests(unittest.TestCase):
    def test_training_report_has_loss_trace_and_rollout_diagnostics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.yaml").write_text(
                "variant: test\nmodel: Qwen/Qwen3-4B\nmax_steps: 50\n"
                "max_completion_length: 4096\n",
                encoding="utf-8",
            )
            (root / "source-commit.txt").write_text("abc\n", encoding="utf-8")
            (root / "runtime-1.txt").write_text("node=node03\nslurm_job_id=1\n", encoding="utf-8")
            summary = {
                "loss_finite": True,
                "loss_history": [
                    {"step": 1, "loss": 0.4, "grad_norm": 0.2},
                    {"step": 2, "loss": 0.1, "grad_norm": 0.3},
                ],
                "vllm_rollout_calls": {
                    "count": 10, "at_completion_cap": 3,
                    "mean_tokens": 100, "max_tokens": 4096, "completion_cap": 4096,
                },
                "recorded_rollouts": {"count": 10, "thinking_closed": 8, "boxed_answer": 7},
                "rollout_dump_integrity": {"covers_latest_checkpoint": True},
                "gpu_telemetry": [{"gpu": 0, "mean_utilization_percent": 92.0, "max_memory_mib": 43000, "max_temperature_c": 60}],
            }
            report = write_training_report(summary, root)
            self.assertAlmostEqual(report["optimization"]["loss_delta"], -0.3)
            self.assertEqual(report["rollouts"]["completion_cap_rate"], 0.3)
            markdown = (root / "report.md").read_text(encoding="utf-8")
            self.assertIn("Optimization trace", markdown)
            self.assertIn("█▁", markdown)
            self.assertTrue((root / "report.json").is_file())

    def test_evaluation_report_copies_official_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = {
                "model": "Qwen/Qwen3-4B", "benchmark": "aime24", "method": "base",
                "samples_per_problem": 12, "avg_at_12": 0.75, "pass_at_12": 0.8,
                "maj_at_12": 0.77, "bootstrap_95ci": [0.6, 0.9],
                "official_format_rate": 0.99, "length_cutoff_rate": 0.01,
            }
            report = write_evaluation_report(summary, root)
            self.assertEqual(report["metrics"]["avg_at_12"], 0.75)
            self.assertIn("Official benchmark metrics", (root / "report.md").read_text(encoding="utf-8"))
            self.assertEqual(json.loads((root / "report.json").read_text())["report_type"], "evaluation")


if __name__ == "__main__":
    unittest.main()
