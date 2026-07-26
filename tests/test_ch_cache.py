import json

from opsd_research.ch_cache import (
    auditor_prompt,
    blind_student_prompt,
    build_dynamic_dossier_record,
)
from opsd_research.ch_dossier import accepted_teacher_dossiers
from opsd_research.merge_ch_cache import merge_ch_cache_shards
from opsd_research.summarize_ch_cache import summarize_ch_records


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
    assert "Blind attempt 1: first" in prompt
    assert "Blind attempt 3: third" in prompt
    assert "Write naturally" in prompt
    assert "Return only one JSON" not in prompt
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


def test_dynamic_dossier_preserves_reference_without_boxed_formatting() -> None:
    record = build_dynamic_dossier_record(
        example_index=9,
        problem="Find the value.",
        reference_solution="A trusted derivation concludes that the value is 7.",
        blind_attempts=("attempt one", "attempt two"),
        audit_text="Attempt one is wrong; attempt two follows the trusted derivation.",
        builder_seed=42,
    )

    assert record["accepted"] is True
    assert record["reference_answer"] == "Established by the trusted reference reasoning"
    assert "trusted derivation concludes" in record["teacher_dossier"]


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


def test_cache_summary_measures_third_attempt_unique_recovery() -> None:
    records = [
        {
            "accepted": True,
            "blind_attempt_count": 3,
            "blind_attempts": [
                "wrong \\boxed{1}", "wrong \\boxed{2}", "right \\boxed{7}"
            ],
            "reference_answer": "7",
            "blind_attempt_output_tokens": [10, 11, 12],
            "audit_output_tokens": 20,
        },
        {
            "accepted": True,
            "blind_attempt_count": 3,
            "blind_attempts": [
                "right \\boxed{5}", "also right \\boxed{5}", "wrong \\boxed{4}"
            ],
            "reference_answer": "5",
            "blind_attempt_output_tokens": [13, 14, 15],
            "audit_output_tokens": 21,
        },
    ]

    summary = summarize_ch_records(records)

    assert summary["accepted_examples"] == 2
    assert summary["third_attempt_unique_recoveries"] == 1
    assert summary["third_attempt_unique_recovery_rate"] == 0.5
    assert summary["correct_rate_by_attempt"] == [0.5, 0.5, 0.5]
    assert summary["total_generation_tokens"] == 116
