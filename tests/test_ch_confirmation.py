from __future__ import annotations

import unittest

from opsd_research.ch_confirmation import validate_confirmation_scale


class ChConfirmationTests(unittest.TestCase):
    def test_complete_diversity_confirmation_scale_is_admitted(self) -> None:
        result = validate_confirmation_scale(
            accepted_indices=range(1682),
            train_indices=range(1600),
            requested_examples=1682,
            required_train_identities=1600,
        )

        self.assertEqual(result["accepted_examples"], 1682)
        self.assertEqual(result["eligible_train_identities"], 1600)
        self.assertEqual(result["required_first_pass_identities"], 1600)

    def test_incomplete_confirmation_scale_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "retain every requested"):
            validate_confirmation_scale(
                accepted_indices=range(1681),
                train_indices=range(1600),
                requested_examples=1682,
                required_train_identities=1600,
            )

        with self.assertRaisesRegex(ValueError, "1600 are required"):
            validate_confirmation_scale(
                accepted_indices=range(1682),
                train_indices=range(1599),
                requested_examples=1682,
                required_train_identities=1600,
            )


if __name__ == "__main__":
    unittest.main()
