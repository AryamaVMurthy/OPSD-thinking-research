from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from opsd_research.training_data import (
    load_math_cot_questions_only,
    partition_indices,
    partition_manifest,
)


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

    def test_problem_only_loader_never_materializes_response_column(self):
        from pyarrow import parquet, table

        with TemporaryDirectory() as directory:
            path = Path(directory) / "data.parquet"
            parquet.write_table(
                table(
                    {
                        "question": ["q1", "q2"],
                        "response": ["private-1", "private-2"],
                        "data_source": ["amc_aime", "aops_forum"],
                    }
                ),
                path,
            )
            with patch(
                "huggingface_hub.hf_hub_download",
                return_value=str(path),
            ):
                loaded = load_math_cot_questions_only()

        self.assertEqual(
            loaded.column_names, ["question", "data_source"]
        )
        self.assertNotIn("response", loaded.column_names)
        self.assertEqual(loaded["question"], ["q1", "q2"])


if __name__ == "__main__":
    unittest.main()
