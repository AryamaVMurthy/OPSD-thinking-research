from opsd_research.build_graf_viability import forced_action_prompt
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
