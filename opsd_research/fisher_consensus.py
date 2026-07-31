"""Answer-free contrastive agreement targets in a frozen Fisher tangent space."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FisherConsensusTarget:
    """Detached trust-region target and the direction used to construct it."""

    target_probs: torch.Tensor
    consensus_direction: torch.Tensor
    metrics: dict[str, torch.Tensor]


@dataclass(frozen=True)
class FisherEdgeBoost:
    """Leave-one-out cross-problem coherence and normalized boost weights."""

    signed_edges: torch.Tensor
    weights: torch.Tensor
    active_fraction: torch.Tensor
    mean_positive_edge: torch.Tensor
    max_positive_edge: torch.Tensor
    signature_norms: torch.Tensor


def _work_dtype(tensor: torch.Tensor) -> torch.dtype:
    return torch.float64 if tensor.dtype == torch.float64 else torch.float32


def fisher_score_signature(
    *,
    base_logits: torch.Tensor,
    consensus_direction: torch.Tensor,
    token_mask: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collapse tokenwise scores into shared Fisher-whitened vocabulary axes."""
    if base_logits.shape != consensus_direction.shape:
        raise ValueError("base logits and consensus direction must match")
    if token_mask.shape != base_logits.shape[:-1]:
        raise ValueError("token_mask must match non-vocabulary dimensions")
    if token_mask.dtype != torch.bool:
        raise ValueError("token_mask must be boolean")
    if base_logits.ndim != 3:
        raise ValueError("Fisher signatures require [batch, token, vocab]")
    if not bool(token_mask.any(dim=1).all().item()):
        raise ValueError("every Fisher signature requires a retained token")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    dtype = _work_dtype(base_logits)
    probability = (
        base_logits.detach().to(dtype) / float(temperature)
    ).softmax(dim=-1)
    whitened = (
        probability.sqrt()
        * consensus_direction.detach().to(dtype)
        * token_mask.unsqueeze(-1)
    )
    counts = token_mask.sum(dim=1, keepdim=True).to(dtype)
    signature = whitened.sum(dim=1) / counts
    norms = signature.norm(dim=-1)
    if not bool(torch.isfinite(signature).all().item()):
        raise ValueError("Fisher signature is nonfinite")
    return signature, norms


def fisher_edge_boost_weights(
    signatures: torch.Tensor,
    *,
    threshold: float,
) -> FisherEdgeBoost:
    """Compute leave-one-out boosting edges without positive self-bias."""
    if signatures.ndim != 2:
        raise ValueError("Fisher edge signatures must be a matrix")
    if signatures.shape[0] < 3:
        raise ValueError("Fisher edge boosting requires at least 3 problems")
    if threshold < 0.0:
        raise ValueError("Fisher edge threshold must be nonnegative")
    if not bool(torch.isfinite(signatures).all().item()):
        raise ValueError("Fisher edge signatures are nonfinite")

    dtype = _work_dtype(signatures)
    values = signatures.detach().to(dtype)
    tiny = torch.finfo(dtype).tiny
    norms = values.norm(dim=-1)
    normalized = torch.where(
        (norms > tiny).unsqueeze(-1),
        values / norms.clamp_min(tiny).unsqueeze(-1),
        torch.zeros_like(values),
    )
    leave_one_out = normalized.sum(dim=0, keepdim=True) - normalized
    leave_norms = leave_one_out.norm(dim=-1)
    leave_unit = torch.where(
        (leave_norms > tiny).unsqueeze(-1),
        leave_one_out
        / leave_norms.clamp_min(tiny).unsqueeze(-1),
        torch.zeros_like(leave_one_out),
    )
    signed_edges = (normalized * leave_unit).sum(dim=-1).clamp(
        min=-1.0,
        max=1.0,
    )
    positive = (signed_edges - float(threshold)).clamp_min(0.0)
    active = positive > 0.0
    positive_mean = positive.mean()
    weights = torch.where(
        positive_mean > tiny,
        positive / positive_mean.clamp_min(tiny),
        torch.zeros_like(positive),
    )
    active_values = positive[active]
    mean_positive = (
        active_values.mean()
        if active_values.numel()
        else torch.zeros((), dtype=dtype, device=values.device)
    )
    max_positive = (
        active_values.max()
        if active_values.numel()
        else torch.zeros((), dtype=dtype, device=values.device)
    )
    return FisherEdgeBoost(
        signed_edges=signed_edges,
        weights=weights,
        active_fraction=active.to(dtype).mean(),
        mean_positive_edge=mean_positive,
        max_positive_edge=max_positive,
        signature_norms=norms,
    )


def fisher_consensus_target(
    *,
    base_logits: torch.Tensor,
    guide_logits: torch.Tensor,
    control_logits: torch.Tensor,
    step_size: float,
    max_target_kl: float,
    temperature: float = 1.0,
    direction_mode: str = "matched_control_residual",
) -> FisherConsensusTarget:
    """Construct a frozen-anchor target from agreement among contrastive guides.

    ``guide_logits`` and ``control_logits`` have a leading pair dimension.
    Every pair compares a problem-specific, answer-free plan with an
    independently shuffled plan.  Logit differences are centered under the
    frozen deploy distribution, which is the categorical Fisher tangent
    representation.  Only the common direction survives:

    ``agreement = ||mean(u_k)||_F^2 / mean(||u_k||_F^2)`` and
    ``consensus = agreement * mean(u_k)``.

    Jensen's inequality bounds agreement in ``[0, 1]``.  Disagreeing plans
    therefore cancel without a correctness label or answer-conditioned view.
    """
    if base_logits.ndim < 2:
        raise ValueError("base_logits must include a vocabulary dimension")
    if guide_logits.shape != control_logits.shape:
        raise ValueError("guide and control logits must have identical shapes")
    if guide_logits.ndim != base_logits.ndim + 1:
        raise ValueError("guide logits require one leading pair dimension")
    if guide_logits.shape[1:] != base_logits.shape:
        raise ValueError("guide/control trailing dimensions must match base")
    if guide_logits.shape[0] < 2:
        raise ValueError("Fisher consensus requires at least two guide pairs")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if step_size < 0.0:
        raise ValueError("step_size must be nonnegative")
    if max_target_kl <= 0.0:
        raise ValueError("max_target_kl must be positive")
    if direction_mode not in {
        "matched_control_residual",
        "positive_plan_barycenter",
    }:
        raise ValueError("unsupported Fisher consensus direction mode")

    dtype = _work_dtype(base_logits)
    anchor = base_logits.detach().to(dtype) / temperature
    probability = anchor.softmax(dim=-1)
    log_probability = anchor.log_softmax(dim=-1)
    if direction_mode == "matched_control_residual":
        contrast = (
            guide_logits.detach().to(dtype)
            - control_logits.detach().to(dtype)
        ) / temperature
    else:
        contrast = (
            guide_logits.detach().to(dtype)
            - base_logits.detach().to(dtype).unsqueeze(0)
        ) / temperature

    pair_probability = probability.unsqueeze(0)
    contrast = contrast - (
        pair_probability * contrast
    ).sum(dim=-1, keepdim=True)
    raw_mean = contrast.mean(dim=0)
    mean_energy = (probability * raw_mean.square()).sum(dim=-1)
    pair_energy = (
        pair_probability * contrast.square()
    ).sum(dim=-1)
    mean_pair_energy = pair_energy.mean(dim=0)
    tiny = torch.finfo(dtype).tiny
    agreement = torch.where(
        mean_pair_energy > tiny,
        mean_energy / mean_pair_energy.clamp_min(tiny),
        torch.zeros_like(mean_pair_energy),
    ).clamp(min=0.0, max=1.0)
    consensus = agreement.unsqueeze(-1) * raw_mean
    consensus = consensus - (
        probability * consensus
    ).sum(dim=-1, keepdim=True)

    def tilted(scale: torch.Tensor):
        target_log = (
            log_probability + scale.unsqueeze(-1) * consensus
        ).log_softmax(dim=-1)
        target = target_log.exp()
        forward_kl = (
            probability * (log_probability - target_log)
        ).sum(dim=-1)
        reverse_kl = (
            target * (target_log - log_probability)
        ).sum(dim=-1)
        return (
            target_log,
            target,
            forward_kl,
            reverse_kl,
            torch.maximum(forward_kl, reverse_kl),
        )

    effective_step = torch.full_like(agreement, float(step_size))
    target_log, target, forward_kl, reverse_kl, worst_kl = tilted(
        effective_step
    )
    needs_clip = worst_kl > float(max_target_kl)
    lower = torch.zeros_like(effective_step)
    upper = effective_step
    for _ in range(32):
        midpoint = (lower + upper) / 2
        *_, midpoint_kl = tilted(midpoint)
        lower = torch.where(
            midpoint_kl <= float(max_target_kl), midpoint, lower
        )
        upper = torch.where(
            midpoint_kl > float(max_target_kl), midpoint, upper
        )
    effective_step = torch.where(needs_clip, lower, effective_step)
    target_log, target, forward_kl, reverse_kl, worst_kl = tilted(
        effective_step
    )
    consensus_energy = (
        probability * consensus.square()
    ).sum(dim=-1)
    pair_cosine_to_mean = (
        (pair_probability * contrast * raw_mean.unsqueeze(0)).sum(dim=-1)
        / (
            pair_energy.clamp_min(tiny).sqrt()
            * mean_energy.unsqueeze(0).clamp_min(tiny).sqrt()
        )
    )

    return FisherConsensusTarget(
        target_probs=target.detach(),
        consensus_direction=consensus.detach(),
        metrics={
            "agreement": agreement.detach(),
            "raw_mean_direction": raw_mean.detach(),
            "mean_direction_energy": mean_energy.detach(),
            "mean_pair_energy": mean_pair_energy.detach(),
            "consensus_energy": consensus_energy.detach(),
            "mean_pair_cosine": pair_cosine_to_mean.mean(dim=0).detach(),
            "min_pair_cosine": pair_cosine_to_mean.min(dim=0).values.detach(),
            "target_forward_kl": forward_kl.detach(),
            "target_reverse_kl": reverse_kl.detach(),
            "target_kl": worst_kl.detach(),
            "effective_step_size": effective_step.detach(),
        },
    )


def fisher_consensus_loss(
    *,
    student_logits: torch.Tensor,
    base_logits: torch.Tensor,
    guide_logits: torch.Tensor,
    control_logits: torch.Tensor,
    token_mask: torch.Tensor,
    step_size: float,
    max_target_kl: float,
    temperature: float = 1.0,
    anchor_kl_weight: float = 0.0,
    direction_mode: str = "matched_control_residual",
) -> tuple[torch.Tensor, FisherConsensusTarget]:
    """Fit the detached target with an optional frozen-policy KL proximal."""
    if student_logits.shape != base_logits.shape:
        raise ValueError("student and base logits must have identical shapes")
    if token_mask.shape != student_logits.shape[:-1]:
        raise ValueError("token_mask must match non-vocabulary dimensions")
    if token_mask.dtype != torch.bool:
        raise ValueError("token_mask must be boolean")
    if not bool(token_mask.any().item()):
        raise ValueError("Fisher consensus loss requires a retained token")
    if anchor_kl_weight < 0.0:
        raise ValueError("anchor_kl_weight must be nonnegative")
    result = fisher_consensus_target(
        base_logits=base_logits,
        guide_logits=guide_logits,
        control_logits=control_logits,
        step_size=step_size,
        max_target_kl=max_target_kl,
        temperature=temperature,
        direction_mode=direction_mode,
    )
    dtype = _work_dtype(student_logits)
    student_log = (
        student_logits.to(dtype) / temperature
    ).log_softmax(dim=-1)
    target = result.target_probs.to(dtype)
    target_log = target.clamp_min(torch.finfo(dtype).tiny).log()
    per_token = (target * (target_log - student_log)).sum(dim=-1)
    base_log = (
        base_logits.detach().to(dtype) / temperature
    ).log_softmax(dim=-1)
    base_probability = base_log.exp()
    student_probability = student_log.exp()
    student_anchor_forward = (
        base_probability * (base_log - student_log)
    ).sum(dim=-1)
    student_anchor_reverse = (
        student_probability * (student_log - base_log)
    ).sum(dim=-1)
    alignment_gain = (
        result.metrics["target_reverse_kl"].to(dtype)
        + student_anchor_forward
        - per_token
    )
    alignment_denominator = 2.0 * (
        result.metrics["target_reverse_kl"].to(dtype)
        * student_anchor_forward
    ).clamp_min(0.0).sqrt()
    alignment_cosine = torch.where(
        alignment_denominator > torch.finfo(dtype).tiny,
        alignment_gain / alignment_denominator,
        torch.zeros_like(alignment_gain),
    ).clamp(min=-1.0, max=1.0)
    optimization_per_token = (
        per_token + float(anchor_kl_weight) * student_anchor_forward
    )
    result.metrics.update(
        {
            "student_anchor_forward_kl": student_anchor_forward.detach(),
            "student_anchor_reverse_kl": student_anchor_reverse.detach(),
            "target_student_kl": per_token.detach(),
            "fisher_alignment_gain": alignment_gain.detach(),
            "fisher_alignment_cosine_proxy": alignment_cosine.detach(),
            "optimization_per_token": optimization_per_token.detach(),
        }
    )
    loss = optimization_per_token[token_mask].mean()
    return loss, result
