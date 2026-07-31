import hashlib

from opsd_research.compare_math import compare_paired_math


def _run(
    method: str,
    checkpoint: str,
    outcomes: list[list[bool]],
    *,
    output_tokens: list[list[int]] | None = None,
    finish_reasons: list[list[str]] | None = None,
):
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
                "output_tokens": (
                    output_tokens[problem_index][sample_index]
                    if output_tokens is not None
                    else 100
                ),
                "finish_reason": (
                    finish_reasons[problem_index][sample_index]
                    if finish_reasons is not None
                    else "stop"
                ),
                "effective_max_new_tokens": 32768,
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
    assert comparison["problem_diagnostics"]["0"] == {
        "baseline_correct": 1,
        "treatment_correct": 2,
        "correct_delta": 1,
        "baseline_pass": 1,
        "treatment_pass": 1,
        "pass_delta": 0,
        "baseline_majority": 1,
        "treatment_majority": 1,
        "majority_delta": 0,
        "improved": 1,
        "degraded": 0,
        "baseline_cutoffs": 0,
        "treatment_cutoffs": 0,
        "cutoff_delta": 0,
        "mean_output_token_delta": 0,
    }
    assert comparison["problem_diagnostics"]["1"]["correct_delta"] == 2
    assert comparison["problem_diagnostics"]["1"]["pass_delta"] == 1


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


def test_paired_math_comparison_localizes_length_and_cutoff_by_flip() -> None:
    baseline_records, baseline_grades = _run(
        "untouched",
        "none",
        [[True, False]],
        output_tokens=[[1000, 32768]],
        finish_reasons=[["stop", "length"]],
    )
    treatment_records, treatment_grades = _run(
        "trained",
        "6",
        [[False, True]],
        output_tokens=[[32768, 2000]],
        finish_reasons=[["length", "stop"]],
    )

    comparison = compare_paired_math(
        baseline_records,
        baseline_grades,
        treatment_records,
        treatment_grades,
        samples_per_problem=2,
        bootstrap_samples=100,
    )

    assert comparison["generation_diagnostics"] == {
        "baseline_mean_output_tokens": 16884.0,
        "treatment_mean_output_tokens": 17384.0,
        "mean_output_token_delta": 500.0,
        "baseline_cutoffs": 1,
        "treatment_cutoffs": 1,
        "cutoff_delta": 0,
    }
    assert comparison["transition_diagnostics"]["degraded"] == {
        "count": 1,
        "mean_output_token_delta": 31768.0,
        "baseline_cutoffs": 0,
        "treatment_cutoffs": 1,
        "cutoff_delta": 1,
    }
    assert comparison["transition_diagnostics"]["improved"] == {
        "count": 1,
        "mean_output_token_delta": -30768.0,
        "baseline_cutoffs": 1,
        "treatment_cutoffs": 0,
        "cutoff_delta": -1,
    }
