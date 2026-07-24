from __future__ import annotations

import unittest
from pathlib import Path

from opsd_research.records import (
    stable_seed,
    validate_adapter_identity,
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


if __name__ == "__main__":
    unittest.main()
