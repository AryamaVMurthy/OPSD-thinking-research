"""Differentiable action-level GRAF losses, independent of a graph builder."""

from __future__ import annotations

import torch
import torch.nn.functional as functional


def branch_routed_loss(
    action_scores: torch.Tensor,
    target_distribution: torch.Tensor,
    fork_mask: torch.Tensor,
    *,
    entropy_floor_fraction: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """KL over meaningful actions plus an optional one-sided entropy floor.

    Shapes are `[batch, actions]` and `[batch]`. Rows without a confident graph
    alignment have `fork_mask=False` and contribute exactly zero gradient.
    """
    if action_scores.ndim != 2 or target_distribution.shape != action_scores.shape:
        raise ValueError("action scores and target distribution must have shape [batch, actions]")
    if fork_mask.shape != action_scores.shape[:1] or fork_mask.dtype != torch.bool:
        raise ValueError("fork_mask must be boolean with shape [batch]")
    if not 0.0 <= entropy_floor_fraction <= 1.0:
        raise ValueError("entropy_floor_fraction must be in [0, 1]")
    active = fork_mask.nonzero(as_tuple=False).flatten()
    if active.numel() == 0:
        zero = action_scores.sum() * 0.0
        return zero, {"branch_kl": zero.detach(), "entropy_floor": zero.detach(), "active_forks": zero.detach()}
    scores = action_scores.index_select(0, active)
    target = target_distribution.index_select(0, active)
    if not torch.allclose(target.sum(dim=-1), torch.ones_like(target[:, 0]), atol=1e-6):
        raise ValueError("each target branch distribution must sum to one")
    log_probs = functional.log_softmax(scores, dim=-1)
    branch_kl = functional.kl_div(log_probs, target, reduction="batchmean")
    probabilities = log_probs.exp()
    student_entropy = -(probabilities * log_probs).sum(dim=-1)
    target_entropy = -(target.clamp_min(1e-12) * target.clamp_min(1e-12).log()).sum(dim=-1)
    entropy_floor = functional.relu(entropy_floor_fraction * target_entropy - student_entropy).square().mean()
    total = branch_kl + entropy_floor
    return total, {
        "branch_kl": branch_kl.detach(),
        "entropy_floor": entropy_floor.detach(),
        "active_forks": torch.tensor(float(active.numel()), device=action_scores.device),
    }
