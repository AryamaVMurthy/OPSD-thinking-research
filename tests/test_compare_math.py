import hashlib

from opsd_research.compare_math import compare_paired_math


def _run(method: str, checkpoint: str, outcomes: list[list[bool]]):
    records = []
    grades = []
    for problem_index, problem_outcomes in enumerate(outcomes):
        for sample_index, correct in enumerate(problem_outcomes):
            response = rf"\boxed{{{10 if correct else 11}}}"
            record = {
                "model": "Qwen/Qwen3-1.7B",
                "method": method,
                "checkpoint": checkpoint,
                "benchmark": "aime25",
                "problem_id": str(problem_index),
                "sample_index": sample_index,
                "seed": problem_index * 10 + sample_index,
                "prompt_hash": f"prompt-{problem_index}",
                "response": response,
                "predicted_answer": "10" if correct else "11",
                "correct": correct,
            }
            records.append(record)
            grades.append(
                {
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
                    "official_predicted_answer": record["predicted_answer"],
                    "official_correct": correct,
                }
            )
    return records, grades


def test_paired_math_comparison_reports_problem_clustered_deltas() -> None:
    baseline_records, baseline_grades = _run(
        "untouched",
        "none",
        [[True, False], [False, False]],
    )
    treatment_records, treatment_grades = _run(
        "opsd-standard-thinking",
        "200",
        [[True, True], [True, True]],
    )

    comparison = compare_paired_math(
        baseline_records,
        baseline_grades,
        treatment_records,
        treatment_grades,
        samples_per_problem=2,
        bootstrap_samples=1000,
    )

    assert comparison["avg_at_2"] == {
        "baseline": 0.25,
        "treatment": 1.0,
        "delta": 0.75,
    }
    assert comparison["pass_at_2"]["delta"] == 0.5
    assert comparison["maj_at_2"]["delta"] == 0.5
    assert comparison["paired_sample_changes"] == {
        "both_correct": 1,
        "both_wrong": 0,
        "degraded": 0,
        "improved": 3,
    }
    assert comparison["pairing_verified"] is True
    assert comparison["evaluation_protocol"] == "development"


def test_explicit_prefix_subset_reuses_first_samples_from_larger_run() -> None:
    baseline_records, baseline_grades = _run(
        "untouched", "none", [[True, False]]
    )
    treatment_records, treatment_grades = _run(
        "trained", "50", [[True, False, True, True]]
    )

    comparison = compare_paired_math(
        baseline_records,
        baseline_grades,
        treatment_records,
        treatment_grades,
        samples_per_problem=2,
        bootstrap_samples=100,
        take_first_samples=True,
    )

    assert comparison["pairing_verified"] is True
    assert comparison["sample_subset_protocol"] == "first-n-paired-samples-v1"
