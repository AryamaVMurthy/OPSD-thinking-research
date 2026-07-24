from __future__ import annotations

import unittest
from pathlib import Path

from opsd_research.records import (
    paired_evaluation_seed,
    stable_seed,
    validate_adapter_identity,
    validate_consistent_fields,
    validate_unique_complete,
)


def record(problem_id: str, sample_index: int):
    return {
        "model": "Qwen/Qwen3-1.7B",
        "method": "untouched",
        "checkpoint": "none",
        "benchmark": "aime25",
        "problem_id": problem_id,
        "sample_index": sample_index,
    }


class RecordTests(unittest.TestCase):
    def test_seed_is_stable_and_sample_specific(self):
        args = (42, "m", "method", "none", "bench", "p")
        self.assertEqual(stable_seed(*args, 0), stable_seed(*args, 0))
        self.assertNotEqual(stable_seed(*args, 0), stable_seed(*args, 1))

    def test_paired_seed_matches_the_accepted_untouched_stream(self):
        expected = stable_seed(
            42,
            "Qwen/Qwen3-1.7B",
            "untouched",
            "none",
            "aime25",
            "7",
            3,
        )
        self.assertEqual(
            paired_evaluation_seed(
                42,
                "Qwen/Qwen3-1.7B",
                "aime25",
                "7",
                3,
            ),
            expected,
        )

    def test_complete_matrix(self):
        records = [record(problem, sample) for problem in ("a", "b") for sample in range(2)]
        validate_unique_complete(records, ("a", "b"), 2)

    def test_duplicate_is_rejected(self):
        records = [record("a", 0), record("a", 0)]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_unique_complete(records, ("a",), 1)

    def test_missing_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_unique_complete([record("a", 0)], ("a",), 2)

    def test_adapter_and_digest_are_coupled(self):
        digest = "a" * 64
        validate_adapter_identity(Path("/adapter"), digest)
        validate_adapter_identity(None, None)
        with self.assertRaisesRegex(ValueError, "together"):
            validate_adapter_identity(Path("/adapter"), None)
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            validate_adapter_identity(Path("/adapter"), "A" * 64)

    def test_consistent_record_identity_is_required(self):
        records = [record("a", 0), record("b", 0)]
        self.assertEqual(
            validate_consistent_fields(records, ("model", "method")),
            {"model": "Qwen/Qwen3-1.7B", "method": "untouched"},
        )
        records[1]["method"] = "different"
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            validate_consistent_fields(records, ("model", "method"))
        with self.assertRaisesRegex(ValueError, "no generation"):
            validate_consistent_fields([], ("model",))


if __name__ == "__main__":
    unittest.main()
