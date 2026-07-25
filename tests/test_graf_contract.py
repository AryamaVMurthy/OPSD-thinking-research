from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from opsd_research.graf_contract import (
    Candidate,
    DevelopmentResult,
    append_ledger,
    qualifies_for_promotion,
    validate_candidate,
)


class GrafContractTests(unittest.TestCase):
    def candidate(self, **kwargs):
        defaults = {
            "candidate_id": "c0-long-2048",
            "parent_id": None,
            "stage": "C0",
            "hypothesis": "Longer trajectories reduce truncation damage.",
            "config": {"max_completion_length": 2048},
            "changed_fields": ("max_completion_length",),
        }
        defaults.update(kwargs)
        return Candidate(**defaults)

    def test_candidate_requires_one_allowlisted_mutation(self):
        validate_candidate(self.candidate(), {"max_completion_length"})
        with self.assertRaisesRegex(ValueError, "exactly one"):
            validate_candidate(
                self.candidate(changed_fields=("max_completion_length", "graph_budget")),
                {"max_completion_length", "graph_budget"},
            )

    def test_promotion_requires_effect_and_positive_interval(self):
        result = DevelopmentResult("c0", 0.031, 0.002, 0.08, 12, "aime24")
        self.assertTrue(qualifies_for_promotion(result, minimum_delta=0.03))
        failed = DevelopmentResult("c0", 0.04, 0.0, 0.08, 12, "aime24")
        self.assertFalse(qualifies_for_promotion(failed, minimum_delta=0.03))

    def test_ledger_is_canonical_jsonl(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "ledger.jsonl"
            digest = append_ledger(path, {"candidate": "c0", "step": 1})
            self.assertEqual(len(digest), 64)
            self.assertEqual(path.read_text(encoding="utf-8"), '{"candidate":"c0","step":1}\n')
