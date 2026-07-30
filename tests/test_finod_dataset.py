from __future__ import annotations

from opsd_research.finod_dataset import finod_training_row


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
