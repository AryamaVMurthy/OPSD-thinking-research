"""Fisher-nuisance orthogonal targets for privileged distillation."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FisherProjectedTarget:
    target_probs: torch.Tensor
    residual_direction: torch.Tensor
    metrics: dict[str, torch.Tensor]


def select_rollout_positions(
    token_mask: torch.Tensor,
    *,
    max_positions: int,
    prefix_tokens: int | None = None,
) -> torch.Tensor:
    """Select deterministic, uniformly spaced positions with any valid token."""
    if token_mask.ndim != 2 or token_mask.dtype != torch.bool:
        raise ValueError("token_mask must be a rank-2 boolean tensor")
    if max_positions <= 0:
        raise ValueError("max_positions must be positive")
    if prefix_tokens is not None and prefix_tokens <= 0:
        raise ValueError("prefix_tokens must be positive when provided")
    candidates = token_mask.any(dim=0).nonzero(as_tuple=False).flatten()
    if prefix_tokens is not None:
        candidates = candidates[candidates < prefix_tokens]
    if not candidates.numel():
        raise ValueError("rollout contains no valid token positions")
    if candidates.numel() <= max_positions:
        return candidates
    offsets = torch.linspace(
        0,
        candidates.numel() - 1,
        steps=max_positions,
        device=candidates.device,
    ).round().to(torch.long)
    return candidates.index_select(0, offsets).unique(sorted=True)


def fisher_projected_loss(
    *,
    student_logits: torch.Tensor,
    guide_logits: torch.Tensor,
    base_logits: torch.Tensor,
    nuisance_logits: torch.Tensor,
    token_mask: torch.Tensor,
    step_size: float,
    nuisance_strength_threshold: float,
    projection_mode: str = "one-sided-positive-v1",
    max_target_kl: float | None = None,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, FisherProjectedTarget]:
    """Forward KL from a detached Fisher-projected target to the student."""
    result = fisher_projected_target(
        student_logits=student_logits,
        guide_logits=guide_logits,
        base_logits=base_logits,
        nuisance_logits=nuisance_logits,
        step_size=step_size,
        nuisance_strength_threshold=nuisance_strength_threshold,
        projection_mode=projection_mode,
        max_target_kl=max_target_kl,
        temperature=temperature,
    )
    if token_mask.shape != student_logits.shape[:-1]:
        raise ValueError("token_mask must match all non-vocabulary dimensions")
    if token_mask.dtype != torch.bool:
        raise ValueError("token_mask must be boolean")
    retained = token_mask.sum()
    if not bool(retained.item()):
        raise ValueError("FiNOD loss requires at least one retained token")
    work_dtype = (
        torch.float64
        if student_logits.dtype == torch.float64
        else torch.float32
    )
    student_log_probs = (
        student_logits.to(work_dtype) / temperature
    ).log_softmax(dim=-1)
    target = result.target_probs.to(work_dtype)
    target_log_probs = target.clamp_min(torch.finfo(work_dtype).tiny).log()
    per_token = (
        target * (target_log_probs - student_log_probs)
    ).sum(dim=-1)
    loss = per_token[token_mask].mean()
    metrics = dict(result.metrics)
    metrics["retained_token_count"] = retained.detach()
    return loss, FisherProjectedTarget(
        target_probs=result.target_probs,
        residual_direction=result.residual_direction,
        metrics=metrics,
    )


def fisher_projected_target(
    *,
    student_logits: torch.Tensor,
    guide_logits: torch.Tensor,
    base_logits: torch.Tensor,
    nuisance_logits: torch.Tensor,
    step_size: float,
    nuisance_strength_threshold: float,
    projection_mode: str = "one-sided-positive-v1",
    max_target_kl: float | None = None,
    temperature: float = 1.0,
) -> FisherProjectedTarget:
    """Build a Fisher-projected exponential-tilt target.

    The guide and nuisance directions are measured relative to the same
    unprivileged frozen-teacher view. ``one-sided-positive-v1`` removes
    positive Fisher alignment and preserves anti-alignment.
    ``signed-orthogonal-v1`` removes the full signed nuisance component.
    ``entropy-neutral-one-sided-v1`` first removes the local
    entropy-gradient component and then applies the one-sided answer filter
    inside that Fisher-orthogonal subspace.
    """
    if not (
        student_logits.shape
        == guide_logits.shape
        == base_logits.shape
        == nuisance_logits.shape
    ):
        raise ValueError("all FiNOD logit tensors must have the same shape")
    if student_logits.ndim < 2:
        raise ValueError("FiNOD logits must have a vocabulary dimension")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if step_size < 0.0:
        raise ValueError("step_size must be nonnegative")
    if nuisance_strength_threshold < 0.0:
        raise ValueError("nuisance_strength_threshold must be nonnegative")
    if projection_mode not in {
        "one-sided-positive-v1",
        "signed-orthogonal-v1",
        "entropy-neutral-one-sided-v1",
    }:
        raise ValueError(f"unsupported FiNOD projection_mode {projection_mode!r}")
    if max_target_kl is not None and max_target_kl <= 0.0:
        raise ValueError("max_target_kl must be positive when provided")

    work_dtype = (
        torch.float64
        if student_logits.dtype == torch.float64
        else torch.float32
    )
    anchor = base_logits.detach().to(work_dtype) / temperature
    probability = anchor.softmax(dim=-1)
    log_probability = anchor.log_softmax(dim=-1)
    guide = (
        guide_logits.detach().to(work_dtype)
        - base_logits.detach().to(work_dtype)
    ) / temperature
    nuisance = (
        nuisance_logits.detach().to(work_dtype)
        - base_logits.detach().to(work_dtype)
    ) / temperature

    def center(direction: torch.Tensor) -> torch.Tensor:
        mean = (probability * direction).sum(dim=-1, keepdim=True)
        return direction - mean

    guide = center(guide)
    nuisance = center(nuisance)
    entropy_direction = center(log_probability)
    guide_energy = (probability * guide.square()).sum(dim=-1)
    nuisance_energy = (probability * nuisance.square()).sum(dim=-1)
    alignment = (probability * guide * nuisance).sum(dim=-1)
    entropy_energy = (
        probability * entropy_direction.square()
    ).sum(dim=-1)
    guide_entropy_alignment = (
        probability * guide * entropy_direction
    ).sum(dim=-1)
    nuisance_entropy_alignment = (
        probability * nuisance * entropy_direction
    ).sum(dim=-1)
    entropy_active = entropy_energy > nuisance_strength_threshold
    entropy_projection_active = torch.zeros_like(entropy_active)
    entropy_coefficient = torch.zeros_like(entropy_energy)
    restricted_guide = guide
    restricted_nuisance = nuisance
    if projection_mode == "entropy-neutral-one-sided-v1":
        entropy_projection_active = entropy_active
        safe_entropy_energy = entropy_energy.clamp_min(
            torch.finfo(work_dtype).tiny
        )
        entropy_coefficient = torch.where(
            entropy_active,
            guide_entropy_alignment / safe_entropy_energy,
            torch.zeros_like(guide_entropy_alignment),
        )
        nuisance_entropy_coefficient = torch.where(
            entropy_active,
            nuisance_entropy_alignment / safe_entropy_energy,
            torch.zeros_like(nuisance_entropy_alignment),
        )
        restricted_guide = center(
            guide - entropy_coefficient.unsqueeze(-1) * entropy_direction
        )
        restricted_nuisance = center(
            nuisance
            - nuisance_entropy_coefficient.unsqueeze(-1)
            * entropy_direction
        )
    restricted_nuisance_energy = (
        probability * restricted_nuisance.square()
    ).sum(dim=-1)
    restricted_alignment = (
        probability * restricted_guide * restricted_nuisance
    ).sum(dim=-1)
    active = restricted_nuisance_energy > nuisance_strength_threshold
    projected_alignment = (
        restricted_alignment
        if projection_mode == "signed-orthogonal-v1"
        else torch.relu(restricted_alignment)
    )
    coefficient = torch.where(
        active,
        projected_alignment
        / restricted_nuisance_energy.clamp_min(
            torch.finfo(work_dtype).tiny
        ),
        torch.zeros_like(restricted_alignment),
    )
    residual = (
        restricted_guide
        - coefficient.unsqueeze(-1) * restricted_nuisance
    )
    residual = center(residual)
    residual_nuisance_alignment = (
        probability * residual * nuisance
    ).sum(dim=-1)
    residual_entropy_alignment = (
        probability * residual * entropy_direction
    ).sum(dim=-1)

    def tilted(
        scale: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        target_log = (
            log_probability + scale.unsqueeze(-1) * residual
        ).log_softmax(dim=-1)
        target_probability = target_log.exp()
        reverse_kl = (
            target_probability * (target_log - log_probability)
        ).sum(dim=-1)
        forward_kl = (
            probability * (log_probability - target_log)
        ).sum(dim=-1)
        worst_kl = torch.maximum(forward_kl, reverse_kl)
        return (
            target_log,
            target_probability,
            forward_kl,
            reverse_kl,
            worst_kl,
        )

    effective_step = torch.full_like(alignment, float(step_size))
    (
        target_log_probs,
        target,
        target_forward_kl,
        target_reverse_kl,
        target_kl,
    ) = tilted(effective_step)
    if max_target_kl is not None:
        needs_clip = target_kl > float(max_target_kl)
        lower = torch.zeros_like(effective_step)
        upper = effective_step
        for _ in range(32):
            midpoint = (lower + upper) / 2
            _, _, _, _, midpoint_kl = tilted(midpoint)
            lower = torch.where(
                midpoint_kl <= float(max_target_kl), midpoint, lower
            )
            upper = torch.where(
                midpoint_kl > float(max_target_kl), midpoint, upper
            )
        effective_step = torch.where(needs_clip, lower, effective_step)
        (
            target_log_probs,
            target,
            target_forward_kl,
            target_reverse_kl,
            target_kl,
        ) = tilted(effective_step)
    residual_energy = (probability * residual.square()).sum(dim=-1)
    student_entropy = -(probability * log_probability).sum(dim=-1)
    target_entropy = -(target * target_log_probs).sum(dim=-1)

    return FisherProjectedTarget(
        target_probs=target.detach(),
        residual_direction=residual.detach(),
        metrics={
            "active_projection": active.detach(),
            "projection_coefficient": coefficient.detach(),
            "entropy_projection_coefficient": (
                entropy_coefficient.detach()
            ),
            "guide_energy": guide_energy.detach(),
            "nuisance_energy": nuisance_energy.detach(),
            "restricted_nuisance_energy": (
                restricted_nuisance_energy.detach()
            ),
            "guide_nuisance_alignment": alignment.detach(),
            "restricted_guide_nuisance_alignment": (
                restricted_alignment.detach()
            ),
            "residual_nuisance_alignment": (
                residual_nuisance_alignment.detach()
            ),
            "entropy_active": entropy_projection_active.detach(),
            "entropy_energy": entropy_energy.detach(),
            "guide_entropy_alignment": (
                guide_entropy_alignment.detach()
            ),
            "residual_entropy_alignment": (
                residual_entropy_alignment.detach()
            ),
            "first_order_entropy_change": (
                -residual_entropy_alignment.detach()
            ),
            "student_entropy": student_entropy.detach(),
            "target_entropy": target_entropy.detach(),
            "target_entropy_change": (
                target_entropy - student_entropy
            ).detach(),
            "residual_energy": residual_energy.detach(),
            "target_forward_kl": target_forward_kl.detach(),
            "target_reverse_kl": target_reverse_kl.detach(),
            "target_kl": target_kl.detach(),
            "effective_step_size": effective_step.detach(),
        },
    )
