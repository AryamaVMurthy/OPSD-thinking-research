import pytest

from opsd_research.compare_lcb import compare_paired_lcb


def _run(method: str, checkpoint: str, outcomes: list[list[bool]]) -> list[dict]:
    records = []
    for problem_index, problem_outcomes in enumerate(outcomes):
        for sample_index, correct in enumerate(problem_outcomes):
            records.append(
                {
                    "model": "Qwen/Qwen3-1.7B",
                    "method": method,
                    "checkpoint": checkpoint,
                    "benchmark": "livecodebench-v6-thinking",
                    "model_revision": "model-revision",
                    "dataset_revision": "dataset-revision",
                    "seed_protocol": "paired-v1",
                    "problem_id": str(problem_index),
                    "sample_index": sample_index,
                    "seed": problem_index * 10 + sample_index,
                    "prompt_hash": f"prompt-{problem_index}",
                    "correct": correct,
                }
            )
    return records


def test_paired_lcb_comparison_reports_problem_clustered_pass_at_k() -> None:
    baseline = _run(
        "untouched",
        "none",
        [[True, False], [False, False]],
    )
    treatment = _run(
        "opsd-standard-thinking",
        "200",
        [[True, True], [True, True]],
    )

    comparison = compare_paired_lcb(
        baseline,
        treatment,
        samples_per_problem=2,
        k_values=(1, 2),
        bootstrap_samples=1000,
    )

    assert comparison["pass_at_1"] == {
        "baseline": 0.25,
        "treatment": 1.0,
        "delta": 0.75,
    }
    assert comparison["pass_at_2"]["delta"] == 0.5
    assert comparison["paired_sample_changes"] == {
        "both_correct": 1,
        "both_wrong": 0,
        "degraded": 0,
        "improved": 3,
    }
    assert comparison["pairing_verified"] is True


def test_paired_lcb_comparison_rejects_dataset_revision_mismatch() -> None:
    baseline = _run("untouched", "none", [[True, False]])
    treatment = _run("opsd-standard-thinking", "200", [[True, False]])
    treatment[0]["dataset_revision"] = "different-revision"

    with pytest.raises(ValueError, match="mismatched dataset_revision"):
        compare_paired_lcb(
            baseline,
            treatment,
            samples_per_problem=2,
            k_values=(1, 2),
            bootstrap_samples=10,
        )
