from __future__ import annotations

import pytest

from opsd_research.finod_dataset import (
    finod_training_row,
    remove_graph_identifiers,
)


def test_finod_row_keeps_answer_control_out_of_student_and_guide_fields():
    row = finod_training_row(
        question="Find the requested integer.",
        reference_solution="A private derivation ends with \\boxed{73}.",
        answer_masked_guide="Check parity, then compare the two boundary cases.",
        source_index=11,
    )

    assert row["problem"] == "Find the requested integer."
    assert row["solution"] == "Check parity, then compare the two boundary cases."
    assert row["finod_answer_control"] == "73"
    assert "73" not in row["problem"]
    assert "73" not in row["solution"]
    assert row["finod_source_index"] == 11


def test_graph_identifiers_cannot_accidentally_reveal_a_short_answer():
    guide = "\n".join(
        [
            "State 4: viable",
            "- 4 [viable]: Isolate n. Check: Verify the inequality.",
        ]
    )

    cleaned = remove_graph_identifiers(guide)
    row = finod_training_row(
        question="Find the requested integer.",
        reference_solution=r"A private derivation ends with \boxed{4}.",
        answer_masked_guide=cleaned,
        source_index=35,
    )

    assert "State 4" not in row["solution"]
    assert "- 4 " not in row["solution"]
    assert "Isolate n" in row["solution"]


def test_semantic_answer_leak_is_still_rejected_after_identifier_removal():
    with pytest.raises(ValueError, match="reference answer"):
        finod_training_row(
            question="Find the requested integer.",
            reference_solution=r"A private derivation ends with \boxed{4}.",
            answer_masked_guide="Test 4 as the final candidate.",
            source_index=35,
        )
