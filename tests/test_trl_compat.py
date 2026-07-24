import unittest
from types import SimpleNamespace

from opsd_research.trl_compat import configure_structured_dataset_args


class StructuredDatasetCompatibilityTests(unittest.TestCase):
    def test_preserves_existing_dataset_kwargs_and_skips_sft_preparation(self):
        args = SimpleNamespace(
            dataset_kwargs={"custom": "value"},
            remove_unused_columns=True,
        )

        returned = configure_structured_dataset_args(args)

        self.assertIs(returned, args)
        self.assertEqual(
            args.dataset_kwargs,
            {"custom": "value", "skip_prepare_dataset": True},
        )
        self.assertFalse(args.remove_unused_columns)

    def test_handles_missing_dataset_kwargs(self):
        args = SimpleNamespace(dataset_kwargs=None, remove_unused_columns=True)

        configure_structured_dataset_args(args)

        self.assertEqual(args.dataset_kwargs, {"skip_prepare_dataset": True})
        self.assertFalse(args.remove_unused_columns)


if __name__ == "__main__":
    unittest.main()
