from __future__ import annotations

from opsd_research.build_fisher_guidance import _plan_prompt


def test_plan_builder_prompt_is_problem_only_and_disables_thinking():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False
            return "</think>\n" + messages[0]["content"]

    prompt = _plan_prompt(
        Tokenizer(), "A private problem statement.", plan_index=1
    )

    assert "A private problem statement." in prompt
    assert "reference solution" not in prompt.lower()
    assert "reported answer" not in prompt.lower()
    assert "do not solve it" in prompt.lower()
