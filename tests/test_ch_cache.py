import json

from opsd_research.ch_cache import (
    auditor_prompt,
    blind_student_prompt,
    build_dossier_record,
    build_dynamic_dossier_record,
)
from opsd_research.ch_dossier import accepted_teacher_dossiers
from opsd_research.merge_ch_cache import merge_ch_cache_shards


def test_blind_student_prompt_has_no_reference_or_answer_context() -> None:
    prompt = blind_student_prompt("Find the integer.", attempt_index=1)
    assert "Find the integer." in prompt
    assert "independent blind attempt 2" in prompt
    assert "reference" not in prompt.lower()
    assert "correct answer" not in prompt.lower()


def test_privileged_auditor_requests_free_form_comparison() -> None:
    prompt = auditor_prompt(
        problem="Find the integer.",
        reference_solution="Verified work gives \\boxed{7}.",
        reference_answer="7",
        attempts=("first", "second", "third"),
    )
    assert "Verified answer: 7" in prompt
    assert "Verified work gives \\boxed{7}." in prompt
    assert "Blind attempt 0: first" in prompt
    assert "Blind attempt 2: third" in prompt
    assert "Write naturally" in prompt
    assert "Return only one JSON" not in prompt


def test_valid_audit_becomes_one_traceable_teacher_dossier() -> None:
    record = build_dossier_record(
        example_index=7,
        problem="Find the integer.",
        reference_solution="Verified work gives \\boxed{7}.",
        blind_attempts=("first", "second", "third"),
        audit_payload={
            "attempts": [
                {
                    "attempt_index": index,
                    "verdict": "correct" if index == 2 else "incorrect",
                    "method": f"method {index}",
                    "first_error": None if index == 2 else f"error {index}",
                    "valid_insights": [],
                    "missing_checks": [] if index == 2 else ["verify"],
                }
                for index in range(3)
            ],
            "best_attempt_index": 2,
            "shared_failure_mode": "the first two attempts skipped verification",
            "corrected_method": ["derive", "verify"],
            "verification_checks": ["substitute the result"],
        },
        builder_seed=42,
    )

    assert record["accepted"] is True
    assert record["example_index"] == 7
    assert record["reference_answer"] == "7"
    assert record["blind_attempt_count"] == 3
    assert record["blind_attempts"] == ["first", "second", "third"]
    assert "Verified answer: 7" in record["teacher_dossier"]


def test_natural_audit_preserves_a_training_example_without_schema_parsing() -> None:
    audit = (
        "The first attempt uses factoring but loses a root, so it is wrong. "
        "The second is partially right but skips the domain check. The third "
        "is correct and verifies both roots. Preserve the factorization, retain "
        "all domain-valid candidates, and substitute each result."
    )
    record = build_dynamic_dossier_record(
        example_index=8,
        problem="Solve the equation.",
        reference_solution="Verified work gives \\boxed{7}.",
        blind_attempts=("free-form one", "free-form two", "free-form three"),
        audit_text=audit,
        builder_seed=42,
    )

    assert record["accepted"] is True
    assert record["audit_format"] == "natural_language_v1"
    assert record["comparative_audit"] == audit
    assert audit in record["teacher_dossier"]


def test_four_complete_shards_merge_into_a_verified_manifest(tmp_path) -> None:
    shards = []
    for shard_id in range(4):
        shard = tmp_path / f"dossiers.shard{shard_id}.jsonl"
        shard.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "accepted": True,
                    "example_index": shard_id,
                    "problem_sha256": f"problem-{shard_id}",
                    "teacher_dossier": f"dossier-{shard_id}",
                    "blind_attempt_count": 3,
                    "num_shards": 4,
                    "shard_id": shard_id,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        shards.append(shard)
    output = tmp_path / "dossiers.jsonl"
    manifest = tmp_path / "manifest.json"

    payload = merge_ch_cache_shards(
        shards=shards,
        output=output,
        manifest=manifest,
        requested_examples=4,
        blind_attempts_per_problem=3,
        builder_model="Qwen/Qwen3-4B",
        builder_model_revision="pinned",
        training_dataset_revision="dataset-pinned",
        builder_seed=42,
    )

    assert payload["accepted_examples"] == 4
    assert payload["rejected_examples"] == 0
    assert accepted_teacher_dossiers(manifest) == {
        index: f"dossier-{index}" for index in range(4)
    }
