from opsd_research.matharena_grading import (
    grade_response,
    rescore_record,
    summarize_rescored,
)


def test_official_grader_distinguishes_factorial_product_from_factorial() -> None:
    grade = grade_response(
        response=r"Therefore the answer is \boxed{26!}.",
        ground_truth=r"2^{25} \cdot 26!",
        strict_parsing=False,
    )

    assert grade["predicted_answer"] == "403291461126605635584000000"
    assert grade["correct"] is False
    assert grade["warning"] == 0


def test_rescore_record_preserves_the_original_grade_in_a_sidecar() -> None:
    original = {
        "model": "Qwen/Qwen3-1.7B",
        "method": "qwen3-instruct",
        "checkpoint": "none",
        "benchmark": "hmmt25",
        "problem_id": "17",
        "sample_index": 3,
        "ground_truth": r"2^{25} \cdot 26!",
        "response": r"Work omitted. \boxed{26!}",
        "correct": True,
    }

    sidecar = rescore_record(original, strict_parsing=False)

    assert original["correct"] is True
    assert sidecar["legacy_correct"] is True
    assert sidecar["official_correct"] is False
    assert sidecar["official_predicted_answer"] == "403291461126605635584000000"
    assert len(sidecar["response_sha256"]) == 64


def test_rescored_summary_uses_only_verified_official_grades() -> None:
    records = []
    for problem_id, ground_truth, predictions in (
        ("0", "10", ("10", "11")),
        ("1", "20", ("20", "20")),
    ):
        for sample_index, prediction in enumerate(predictions):
            records.append(
                {
                    "model": "Qwen/Qwen3-1.7B",
                    "model_revision": "model-revision",
                    "method": "qwen3-instruct",
                    "checkpoint": "none",
                    "adapter_sha256": None,
                    "benchmark": "aime25",
                    "dataset_revision": "dataset-revision",
                    "seed_protocol": "paired-to-untouched-v1",
                    "problem_id": problem_id,
                    "sample_index": sample_index,
                    "seed": sample_index,
                    "prompt_hash": f"prompt-{problem_id}",
                    "ground_truth": ground_truth,
                    "response": rf"\boxed{{{prediction}}}",
                    "predicted_answer": prediction,
                    "correct": True,
                    "formatted": True,
                    "nonempty_thinking": True,
                    "output_tokens": 10,
                    "finish_reason": "stop",
                }
            )
    sidecars = [rescore_record(record, strict_parsing=False) for record in records]

    summary = summarize_rescored(
        records,
        sidecars,
        samples_per_problem=2,
        grader_revision="matharena-revision",
        strict_parsing=False,
    )

    assert summary["avg_at_2"] == 0.75
    assert summary["pass_at_2"] == 1.0
    assert summary["maj_at_2"] == 1.0
    assert summary["grade_changes"] == {
        "changed": 1,
        "legacy_false_negative": 0,
        "legacy_false_positive": 1,
    }


def test_official_grader_supports_algebraic_latex_answers() -> None:
    answer = r"\frac{9 \sqrt{23}}{23}"

    grade = grade_response(
        response=rf"\boxed{{{answer}}}",
        ground_truth=answer,
        strict_parsing=False,
    )

    assert grade["correct"] is True


def test_official_grader_bounds_unrenderable_symbolic_answers() -> None:
    grade = grade_response(
        response=r"\boxed{\frac{2016}{2025!}}",
        ground_truth=r"\frac{1311}{2017}",
        strict_parsing=False,
    )

    assert grade["correct"] is False
    assert grade["predicted_answer"].startswith("<unrenderable:")
    assert len(grade["predicted_answer"]) < 128
