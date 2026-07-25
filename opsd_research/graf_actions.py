"""Canonical assistant-side textual realization of a GRAF graph action."""

from __future__ import annotations


# This string is intentionally shared by viability construction and the
# differentiable action scorer.  A branch target is valid only if both phases
# condition on the exact same assistant-side action prefix.
ACTION_PREFIX = "\n\nBegin your solution by carrying out this proposed mathematical action: "
ACTION_SUFFIX = "\n"
ASSISTANT_ACTION_PREFIX_PROTOCOL = "assistant_action_continuation_v1"


def action_continuation(description: str) -> str:
    """Return the exact model continuation representing a canonical action."""
    return f"{ACTION_PREFIX}{description}{ACTION_SUFFIX}"
