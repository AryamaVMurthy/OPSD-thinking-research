import pytest

from opsd_research.build_graf_cache import (
    MAX_COMPLETION_TOKENS,
    MAX_MODEL_LEN,
    _builder_chat_prompt,
    _cache_builder_chat_prompt,
    _critic_chat_prompt,
    _representative_source_indices,
    _sanitizer_chat_prompt,
    _bounded_builder_prompt,
    _json_object,
)


class _CharacterTokenizer:
    def encode(self, text, *, add_special_tokens=False):
        del add_special_tokens
        return list(text)

    def decode(self, token_ids, *, skip_special_tokens=True):
        del skip_special_tokens
        return "".join(token_ids)

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert not tokenize
        assert add_generation_prompt
        assert not enable_thinking
        return "<assistant><think>\n\n</think>\n\n" + messages[0]["content"]


def test_extracts_plain_or_fenced_json_object() -> None:
    assert _json_object('{"forks": []}') == {"forks": []}
    assert _json_object('```json\n{"forks": []}\n```') == {"forks": []}


def test_rejects_non_json_builder_response() -> None:
    with pytest.raises(ValueError, match="JSON"):
        _json_object("I cannot provide that.")


def test_builder_prompt_bounds_an_unusually_long_reference() -> None:
    prompt = _bounded_builder_prompt(
        _CharacterTokenizer(), "Find x.", "a" * (MAX_MODEL_LEN * 2)
    )
    assert len(prompt) <= MAX_MODEL_LEN - MAX_COMPLETION_TOKENS
    assert "Reference truncated" in prompt


def test_builder_uses_qwen_non_thinking_chat_template_for_json() -> None:
    prompt = _builder_chat_prompt(_CharacterTokenizer(), "Find x.", "A short reference.")
    assert "</think>" in prompt
    assert "Output only the JSON object" in prompt


def test_sanitizer_never_receives_reference_solution() -> None:
    prompt = _sanitizer_chat_prompt(_CharacterTokenizer(), "Find x.")
    assert "Construct a fresh" in prompt
    assert "Candidate graph" not in prompt
    assert "reference solution" not in prompt.lower()


def test_answer_blind_cache_builder_never_receives_reference_solution() -> None:
    prompt = _cache_builder_chat_prompt(
        _CharacterTokenizer(),
        "Find x.",
        "PRIVATE REFERENCE ANSWER",
        answer_blind=True,
    )

    assert "Find x." in prompt
    assert "PRIVATE REFERENCE ANSWER" not in prompt
    assert "Construct a fresh" in prompt


def test_critic_is_privileged_but_requires_answer_masked_json_revision() -> None:
    prompt = _critic_chat_prompt(
        _CharacterTokenizer(), "Find x.", "Private reference reasoning.",
        {"forks": [{"fork_id": "f", "state": "start", "actions": []}]},
    )
    assert "Private reference reasoning." in prompt
    assert "do not copy it into JSON" in prompt
    assert "Return JSON with only `forks`" in prompt


def test_representative_hash_selection_is_deterministic_and_shard_complete() -> None:
    rows = [
        {"question": f"problem {index}", "response": f"solution {index}"}
        for index in range(100)
    ]

    selected = _representative_source_indices(rows, limit=40, seed=73)
    shards = [
        _representative_source_indices(
            rows,
            limit=40,
            seed=73,
            shard_id=shard,
            num_shards=4,
        )
        for shard in range(4)
    ]

    assert len(selected) == len(set(selected)) == 40
    assert selected != list(range(40))
    assert sorted(index for shard in shards for index in shard) == sorted(selected)
    assert all(len(shard) == 10 for shard in shards)
    assert _representative_source_indices(rows, limit=40, seed=74) != selected
