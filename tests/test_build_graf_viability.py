from opsd_research.build_graf_viability import forced_action_messages


def test_forced_action_keeps_the_normal_math_prompt_and_adds_action() -> None:
    messages = forced_action_messages("Solve for x.", "factor the polynomial")

    assert len(messages) == 1
    assert "Solve for x." in messages[0]["content"]
    assert "factor the polynomial" in messages[0]["content"]
    assert "\\boxed{}" in messages[0]["content"]
