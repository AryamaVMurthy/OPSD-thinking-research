import pytest

from opsd_research.build_graf_cache import (
    MAX_COMPLETION_TOKENS,
    MAX_MODEL_LEN,
    _builder_chat_prompt,
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
    prompt = _sanitizer_chat_prompt(
        _CharacterTokenizer(), "Find x.", {"forks": []}
    )
    assert "Candidate graph" in prompt
    assert "reference solution" not in prompt.lower()
