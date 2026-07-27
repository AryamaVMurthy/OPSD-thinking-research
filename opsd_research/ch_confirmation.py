"""Evidence gates for diversity-scaled contrastive-hindsight confirmation."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any


def validate_confirmation_scale(
    *,
    accepted_indices: Collection[int],
    train_indices: Collection[int],
    requested_examples: int,
    required_train_identities: int,
) -> dict[str, Any]:
    """Admit only a complete cache with one non-recycled training pass."""
    if requested_examples < 1 or required_train_identities < 1:
        raise ValueError("confirmation scale requirements must be positive")
    accepted = {int(index) for index in accepted_indices}
    expected = set(range(requested_examples))
    if accepted != expected:
        raise ValueError(
            "confirmation cache must retain every requested source identity"
        )
    eligible = len(accepted.intersection(int(index) for index in train_indices))
    if eligible < required_train_identities:
        raise ValueError(
            f"only {eligible} train identities remain; "
            f"{required_train_identities} are required for one non-recycled pass"
        )
    return {
        "event": "ch1_diversity_confirmation_scale_gate",
        "requested_examples": requested_examples,
        "accepted_examples": len(accepted),
        "eligible_train_identities": eligible,
        "required_first_pass_identities": required_train_identities,
    }
