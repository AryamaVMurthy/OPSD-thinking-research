import pytest

from opsd_research.build_ch_cache_shard import _chat_prompt


class _CharacterTokenizer:
    def encode(self, text, *, add_special_tokens=False):
        del add_special_tokens
        return list(text)

    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, enable_thinking
    ):
        del enable_thinking
        assert not tokenize
        assert add_generation_prompt
        return messages[0]["content"]


def test_ch_prompt_uses_candidate_context_budget() -> None:
    prompt = _chat_prompt(
        _CharacterTokenizer(),
        "short",
        enable_thinking=True,
        completion_tokens=4096,
        max_model_len=4101,
    )
    assert prompt == "short"

    with pytest.raises(ValueError, match="4100"):
        _chat_prompt(
            _CharacterTokenizer(),
            "short",
            enable_thinking=True,
            completion_tokens=4096,
            max_model_len=4100,
        )
