import hashlib
import json
from pathlib import Path

from opsd_research.rescore_matharena import rescore_run


def test_rescore_run_writes_verified_sidecars_without_touching_raw_data(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "generations.jsonl"
    records = []
    for problem_index in range(30):
        answer = problem_index + 1
        for sample_index in range(12):
            records.append(
                {
                    "model": "Qwen/Qwen3-1.7B",
                    "model_revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
                    "method": "qwen3-instruct",
                    "checkpoint": "none",
                    "adapter_sha256": None,
                    "benchmark": "aime25",
                    "dataset_revision": "c94da77eb22bbd6439e62a323bec18493a421302",
                    "problem_id": str(problem_index),
                    "sample_index": sample_index,
                    "seed": sample_index,
                    "prompt_hash": f"prompt-{problem_index}",
                    "ground_truth": str(answer),
                    "response": rf"\boxed{{{answer}}}",
                    "predicted_answer": str(answer),
                    "correct": True,
                    "formatted": True,
                    "nonempty_thinking": True,
                    "output_tokens": 10,
                    "finish_reason": "stop",
                }
            )
    raw_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    original_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    grades_path = tmp_path / "official-grades.jsonl"
    summary_path = tmp_path / "summary.official.json"

    rescore_run(
        config_path=Path(
            "reproductions/01_qwen3_thinking_math/configs/qwen3-1p7b-aime25.yaml"
        ),
        input_paths=[raw_path],
        grades_output=grades_path,
        summary_output=summary_path,
    )

    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == original_hash
    assert len(grades_path.read_text(encoding="utf-8").splitlines()) == 360
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["avg_at_12"] == 1.0
    assert summary["num_problems"] == 30
    assert summary["grader_revision"] == "a11194deff8c67a232974a383795e8a2776b4c6f"
    assert summary["seed_protocol"] is None
    assert summary["scorer_environment"]["antlr4-python3-runtime"] == "4.11.0"
