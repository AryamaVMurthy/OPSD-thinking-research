import unittest

from opsd_research.graf_development_report import render


class GrafDevelopmentReportTests(unittest.TestCase):
    def test_renders_avg12_and_direct_delta_decision(self):
        comparison = {
            "model": "Qwen/Qwen3-4B", "benchmark": "aime24",
            "comparison_protocol": "paired-problem-cluster-bootstrap-v1",
            "num_problems": 30, "samples_per_problem": 12,
            "avg_at_12": {"baseline": .5, "treatment": .55, "delta": .05},
            "delta_bootstrap_95ci": {"avg_at_12": [-.01, .11]},
            "paired_sample_changes": {"improved": 10, "degraded": 8, "both_correct": 20, "both_wrong": 322},
        }
        report = render(comparison, candidate_id="g4")
        self.assertIn("+5.00 pp", report)
        self.assertIn("promote to full confirmation", report)
        self.assertIn("interpretation only", report)
