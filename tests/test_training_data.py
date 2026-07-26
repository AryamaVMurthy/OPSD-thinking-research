from __future__ import annotations

import unittest

from opsd_research.training_data import partition_indices, partition_manifest


class TrainingDataTests(unittest.TestCase):
    def test_content_partition_is_stable_and_complete(self):
        dataset = [
            {"question": f"q-{index}", "response": f"r-{index}"}
            for index in range(100)
        ]
        train_one, heldout_one = partition_indices(dataset, 0.2)
        train_two, heldout_two = partition_indices(dataset, 0.2)
        self.assertEqual(train_one, train_two)
        self.assertEqual(heldout_one, heldout_two)
        self.assertEqual(set(train_one).intersection(heldout_one), set())
        self.assertEqual(set(train_one).union(heldout_one), set(range(100)))
        manifest = partition_manifest(dataset, 0.2)
        self.assertEqual(manifest["train_examples"], len(train_one))
        self.assertEqual(manifest["heldout_diagnostic_examples"], len(heldout_one))
        self.assertEqual(len(str(manifest["heldout_content_sha256"])), 64)

    def test_zero_fraction_preserves_all_training_rows(self):
        dataset = [{"question": "q", "response": "r"}]
        self.assertEqual(partition_indices(dataset, 0.0), ([0], []))


if __name__ == "__main__":
    unittest.main()
