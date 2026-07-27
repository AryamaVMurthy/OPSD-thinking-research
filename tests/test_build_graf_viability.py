import hashlib
import json

import pytest

from opsd_research.build_graf_viability import (
    _require_routable_forks,
    forced_action_prompt,
    reference_answer_from_solution,
    viability_trial_seed,
    viability_trial_record,
)
from opsd_research.graf_actions import action_continuation


class TinyTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is False
        assert add_generation_prompt is True
        if enable_thinking:
            return f"<assistant-think>{messages[0]['content']}"
        return "<assistant></think>"


def test_forced_action_keeps_the_normal_math_prompt_and_appends_assistant_action() -> None:
    prompt = forced_action_prompt(TinyTokenizer(), "Solve for x.", "factor the polynomial")

    assert "Solve for x." in prompt
    assert "\\boxed{}" in prompt
    assert prompt.endswith(action_continuation("factor the polynomial"))


def test_viability_preflight_rejects_a_non_routable_cached_fork() -> None:
    records = [{
        "example_index": 12,
        "graph": {"forks": [{
            "fork_id": "dead-fork",
            "actions": [{"status": "invalid"}, {"status": "dead_end"}],
        }]},
    }]
    try:
        _require_routable_forks(records)
    except SystemExit as error:
        assert "example_index=12" in str(error)
        assert "dead-fork" in str(error)
    else:
        raise AssertionError("non-routable cache record was accepted")


def test_viability_trial_record_retains_regradable_generation_evidence() -> None:
    record = viability_trial_record(
        example_index=12,
        fork_id="fork-a",
        action_id="action-b",
        sample_index=1,
        seed=4201,
        prompt="student prompt plus forced action",
        reference_answer=r"\frac{1}{2}",
        completion=r"Reasoning...\n\boxed{\frac{1}{2}}",
        output_token_ids=[10, 20, 30],
        finish_reason="stop",
        legacy_correct=True,
        model="Qwen/Qwen3-4B",
        model_revision="revision",
        forced_prefix_protocol="assistant_action_continuation_v1",
    )

    assert record["schema_version"] == 1
    assert record["raw_completion"] == r"Reasoning...\n\boxed{\frac{1}{2}}"
    assert record["extracted_answer"] == r"\frac{1}{2}"
    assert record["reference_answer"] == r"\frac{1}{2}"
    assert record["generation_seed"] == 4201
    assert record["output_tokens"] == 3
    assert record["finish_reason"] == "stop"
    assert record["legacy_grader"] == "generation_common.grade_math@v1"
    assert record["legacy_correct"] is True
    assert record["prompt_sha256"] == hashlib.sha256(
        b"student prompt plus forced action"
    ).hexdigest()
    assert record["completion_sha256"] == hashlib.sha256(
        record["raw_completion"].encode()
    ).hexdigest()
    assert record["output_token_ids_sha256"] == hashlib.sha256(
        json.dumps([10, 20, 30], separators=(",", ":")).encode()
    ).hexdigest()


def test_viability_trial_seed_is_identity_stable_and_sample_specific() -> None:
    first = viability_trial_seed(42, 12, "fork-a", "action-b", 0)

    assert first == viability_trial_seed(42, 12, "fork-a", "action-b", 0)
    assert first != viability_trial_seed(42, 12, "fork-a", "action-b", 1)
    assert first != viability_trial_seed(42, 12, "fork-a", "action-c", 0)
    assert 0 <= first <= 0x7FFFFFFF


def test_reference_answer_is_extracted_from_the_worked_solution() -> None:
    solution = r"First derive the invariant. Therefore \boxed{\frac{7}{3}}."

    assert reference_answer_from_solution(solution) == r"\frac{7}{3}"


def test_reference_answer_rejects_an_unverifiable_worked_solution() -> None:
    with pytest.raises(ValueError, match="final boxed answer"):
        reference_answer_from_solution("A plausible derivation without a final target")
