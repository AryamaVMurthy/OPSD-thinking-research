"""Canonical assistant-side textual realization of a GRAF graph action."""

from __future__ import annotations


# This string is intentionally shared by viability construction and the
# differentiable action scorer.  A branch target is valid only if both phases
# condition on the exact same assistant-side action prefix.
ACTION_PREFIX = "\n\nBegin your solution by carrying out this proposed mathematical action: "
ACTION_SUFFIX = "\n"
ASSISTANT_ACTION_PREFIX_PROTOCOL = "assistant_action_continuation_v1"
RECOVERY_ACTION_PREFIX_PROTOCOL = "assistant_recovery_conditioned_continuation_v1"


def action_continuation(description: str) -> str:
    """Return the exact model continuation representing a canonical action."""
    return f"{ACTION_PREFIX}{description}{ACTION_SUFFIX}"


def recovery_conditioned_description(
    description: str, validation_test: str, recovery_action: str
) -> str:
    """Render an action that explicitly verifies and repairs its own branch.

    This is a candidate continuation, not an extra teacher prompt.  The same
    text is used for forced-continuation viability estimation and the student
    policy score, making the recovery behavior measurable rather than a
    decorative graph annotation.
    """
    return (
        f"{description}\n"
        f"Before committing, verify: {validation_test}\n"
        f"If that check fails, explicitly retract this route and instead: {recovery_action}"
    )
