import hashlib

import pytest

from opsd_research.review_math_changes import build_paired_review


def _record(
    *,
    method: str,
    checkpoint: str,
    problem_id: str,
    sample_index: int,
    correct: bool,
) -> tuple[dict, dict]:
    answer = "10" if correct else "11"
    response = rf"<think>reasoning {answer}</think>\boxed{{{answer}}}"
    record = {
        "model": "Qwen/Qwen3-4B",
        "method": method,
        "checkpoint": checkpoint,
        "benchmark": "aime25",
        "problem_id": problem_id,
        "sample_index": sample_index,
        "seed": 100 + sample_index,
        "prompt_hash": f"prompt-{problem_id}",
        "problem": f"Find the answer to {problem_id}.",
        "ground_truth": "10",
        "response": response,
        "reasoning": f"reasoning {answer}",
        "final": rf"\boxed{{{answer}}}",
        "predicted_answer": answer,
        "correct": not correct,
        "output_tokens": 20,
        "finish_reason": "stop",
    }
    grade = {
        **{
            key: record[key]
            for key in (
                "model",
                "method",
                "checkpoint",
                "benchmark",
                "problem_id",
                "sample_index",
            )
        },
        "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
        "official_predicted_answer": answer,
        "official_correct": correct,
    }
    return record, grade


def test_paired_review_surfaces_all_outcome_classes() -> None:
    outcomes = [
        (True, False),
        (False, True),
        (False, False),
        (True, True),
    ]
    baseline_records = []
    baseline_grades = []
    treatment_records = []
    treatment_grades = []
    for index, (base_correct, treatment_correct) in enumerate(outcomes):
        base, base_grade = _record(
            method="untouched",
            checkpoint="none",
            problem_id=str(index),
            sample_index=index,
            correct=base_correct,
        )
        treatment, treatment_grade = _record(
            method="opsd-standard-thinking",
            checkpoint="50",
            problem_id=str(index),
            sample_index=index,
            correct=treatment_correct,
        )
        baseline_records.append(base)
        baseline_grades.append(base_grade)
        treatment_records.append(treatment)
        treatment_grades.append(treatment_grade)

    packet = build_paired_review(
        baseline_records,
        baseline_grades,
        treatment_records,
        treatment_grades,
        per_class=1,
    )

    assert "## Degraded" in packet
    assert "## Improved" in packet
    assert "## Both Wrong" in packet
    assert "## Both Correct" in packet
    assert packet.count("Total paired samples in class: `1`") == 4
    assert packet.count("### Untouched baseline") == 4
    assert packet.count("### OPSD treatment") == 4
    assert "Officially correct: `True`" in packet
    assert "Officially correct: `False`" in packet


def test_paired_review_rejects_seed_mismatch() -> None:
    base, base_grade = _record(
        method="untouched",
        checkpoint="none",
        problem_id="0",
        sample_index=0,
        correct=True,
    )
    treatment, treatment_grade = _record(
        method="opsd-standard-thinking",
        checkpoint="50",
        problem_id="0",
        sample_index=0,
        correct=False,
    )
    treatment["seed"] += 1

    with pytest.raises(ValueError, match="mismatched seed"):
        build_paired_review(
            [base],
            [base_grade],
            [treatment],
            [treatment_grade],
        )
