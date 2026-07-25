import hashlib

from opsd_research.review_rollouts import apply_official_grades


def test_review_selection_uses_hash_verified_official_grades() -> None:
    response = r"\boxed{26!}"
    raw = {
        "model": "Qwen/Qwen3-1.7B",
        "method": "opsd-standard-thinking",
        "checkpoint": "100",
        "benchmark": "hmmt25",
        "problem_id": "17",
        "sample_index": 1,
        "response": response,
        "predicted_answer": "26!",
        "correct": True,
    }
    official = {
        "model": raw["model"],
        "method": raw["method"],
        "checkpoint": raw["checkpoint"],
        "benchmark": raw["benchmark"],
        "problem_id": raw["problem_id"],
        "sample_index": raw["sample_index"],
        "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
        "official_predicted_answer": "403291461126605635584000000",
        "official_correct": False,
    }

    merged = apply_official_grades([raw], [official])

    assert raw["correct"] is True
    assert merged[0]["legacy_correct"] is True
    assert merged[0]["correct"] is False
    assert merged[0]["predicted_answer"] == "403291461126605635584000000"
    assert merged[0]["review_grading_protocol"] == "official-matharena-parser-v1"
