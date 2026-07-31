"""Answer-free contrastive agreement targets in a frozen Fisher tangent space."""

from __future__ import annotations

import math
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
    retraction_mode: str = "exponential",
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
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if not math.isfinite(step_size) or step_size < 0.0:
        raise ValueError("step_size must be finite and nonnegative")
    if not math.isfinite(max_target_kl) or max_target_kl <= 0.0:
        raise ValueError("max_target_kl must be finite and positive")
    if direction_mode not in {
        "matched_control_residual",
        "positive_plan_barycenter",
        "entropy_neutral_plan_barycenter",
    }:
        raise ValueError("unsupported Fisher consensus direction mode")
    if retraction_mode not in {
        "exponential",
        "self_information_mixture",
        "self_information_exponential",
    }:
        raise ValueError("unsupported Fisher consensus retraction mode")
    if (
        retraction_mode
        in {"self_information_mixture", "self_information_exponential"}
        and direction_mode != "entropy_neutral_plan_barycenter"
    ):
        raise ValueError(
            "self-information retraction requires entropy-neutral direction"
        )
    if not bool(torch.isfinite(base_logits).all().item()):
        raise ValueError("base logits are nonfinite")
    if not bool(torch.isfinite(guide_logits).all().item()):
        raise ValueError("guide logits are nonfinite")
    if direction_mode == "matched_control_residual" and not bool(
        torch.isfinite(control_logits).all().item()
    ):
        raise ValueError("control logits are nonfinite")

    dtype = _work_dtype(base_logits)
    preserves_self_information = retraction_mode in {
        "self_information_mixture",
        "self_information_exponential",
    }
    anchor = base_logits.detach().to(dtype) / temperature
    probability = anchor.softmax(dim=-1)
    if preserves_self_information:
        probability = probability / probability.sum(
            dim=-1,
            keepdim=True,
            dtype=torch.float64,
        ).to(dtype)
    log_probability = anchor.log_softmax(dim=-1)

    def weighted_sum(
        weight: torch.Tensor,
        value: torch.Tensor,
        *,
        keepdim: bool = False,
    ) -> torch.Tensor:
        product = weight * value
        if preserves_self_information:
            return product.sum(
                dim=-1,
                keepdim=keepdim,
                dtype=torch.float64,
            ).to(dtype)
        return product.sum(dim=-1, keepdim=keepdim)

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
    contrast = contrast - weighted_sum(
        pair_probability,
        contrast,
        keepdim=True,
    )
    raw_mean = contrast.mean(dim=0)
    mean_energy = weighted_sum(probability, raw_mean.square())
    pair_energy = weighted_sum(pair_probability, contrast.square())
    mean_pair_energy = pair_energy.mean(dim=0)
    tiny = torch.finfo(dtype).tiny
    agreement = torch.where(
        mean_pair_energy > tiny,
        mean_energy / mean_pair_energy.clamp_min(tiny),
        torch.zeros_like(mean_pair_energy),
    ).clamp(min=0.0, max=1.0)
    consensus = agreement.unsqueeze(-1) * raw_mean
    consensus = consensus - weighted_sum(
        probability,
        consensus,
        keepdim=True,
    )
    entropy_direction = log_probability - (
        weighted_sum(probability, log_probability, keepdim=True)
    )
    entropy_energy = weighted_sum(probability, entropy_direction.square())
    entropy_alignment_before = weighted_sum(
        probability,
        consensus * entropy_direction,
    )
    consensus_energy_before_projection = weighted_sum(
        probability,
        consensus.square(),
    )
    if direction_mode == "entropy_neutral_plan_barycenter":
        entropy_active = entropy_energy > torch.finfo(dtype).eps
        entropy_coefficient = torch.where(
            entropy_active,
            entropy_alignment_before
            / entropy_energy.clamp_min(torch.finfo(dtype).eps),
            torch.zeros_like(entropy_alignment_before),
        )
        projected_consensus = consensus - (
            entropy_coefficient.unsqueeze(-1) * entropy_direction
        )
        projected_consensus = projected_consensus - weighted_sum(
            probability,
            projected_consensus,
            keepdim=True,
        )
        consensus = torch.where(
            entropy_active.unsqueeze(-1),
            projected_consensus,
            consensus,
        )
    entropy_alignment_after = weighted_sum(
        probability,
        consensus * entropy_direction,
    )
    consensus_energy_after_projection = weighted_sum(
        probability,
        consensus.square(),
    )
    if direction_mode == "entropy_neutral_plan_barycenter":
        numerical_scale = 128.0 * torch.finfo(dtype).eps
        centering_residual = weighted_sum(
            probability,
            consensus,
        ).abs()
        centering_tolerance = numerical_scale * (
            1.0 + consensus_energy_after_projection.sqrt()
        )
        alignment_tolerance = numerical_scale * (
            1.0
            + (
                entropy_energy * consensus_energy_after_projection
            ).clamp_min(0.0).sqrt()
        )
        projection_values = torch.stack(
            [
                entropy_energy,
                entropy_alignment_before,
                entropy_alignment_after,
                consensus_energy_after_projection,
                centering_residual,
            ]
        )
        if not bool(torch.isfinite(projection_values).all().item()):
            raise RuntimeError("entropy-neutral Fisher projection is nonfinite")
        if bool((centering_residual > centering_tolerance).any().item()):
            raise RuntimeError("entropy-neutral Fisher direction is uncentered")
        if bool(
            (
                entropy_active
                & (entropy_alignment_after.abs() > alignment_tolerance)
            ).any().item()
        ):
            raise RuntimeError(
                "entropy-neutral Fisher projection exceeds alignment tolerance"
            )
    retained_direction_energy_fraction = torch.where(
        consensus_energy_before_projection > tiny,
        consensus_energy_after_projection
        / consensus_energy_before_projection.clamp_min(tiny),
        torch.ones_like(consensus_energy_before_projection),
    ).clamp(min=0.0, max=1.0)

    requested_step = torch.full_like(agreement, float(step_size))
    positivity_limited = torch.zeros_like(agreement, dtype=torch.bool)
    if retraction_mode == "self_information_mixture":
        positivity_bound = torch.where(
            consensus < 0.0,
            -1.0 / consensus,
            torch.full_like(consensus, float("inf")),
        ).amin(dim=-1)
        positivity_margin = math.sqrt(torch.finfo(dtype).eps)
        positivity_upper = positivity_bound * (1.0 - positivity_margin)
        effective_step = torch.minimum(requested_step, positivity_upper)
        positivity_limited = effective_step < requested_step
    else:
        effective_step = requested_step

    def corrected_exponential(scale: torch.Tensor):
        """Return the self-information I-projection at a fixed tangent scale."""
        beta = torch.zeros_like(scale)
        moment_tolerance = max(1e-12, 32.0 * torch.finfo(dtype).eps)

        def distribution(multiplier: torch.Tensor):
            corrected_log = (
                log_probability
                + scale.unsqueeze(-1) * consensus
                + multiplier.unsqueeze(-1) * entropy_direction
            ).log_softmax(dim=-1)
            corrected = corrected_log.exp()
            corrected_normalizer = corrected.sum(
                dim=-1,
                keepdim=True,
                dtype=torch.float64,
            ).to(dtype)
            corrected = corrected / corrected_normalizer
            corrected_log = corrected_log - corrected_normalizer.log()
            moment = weighted_sum(corrected, entropy_direction)
            centered_information = entropy_direction - moment.unsqueeze(-1)
            variance = weighted_sum(
                corrected,
                centered_information.square(),
            )
            return corrected_log, corrected, moment, variance

        # The scalar moment is monotone in beta. Newton converges rapidly in
        # the small KL ball; bounded updates avoid a rare skewed-tail jump.
        for _ in range(12):
            _, _, moment, variance = distribution(beta)
            update = torch.where(
                entropy_active,
                moment / variance.clamp_min(torch.finfo(dtype).eps),
                torch.zeros_like(moment),
            ).clamp(min=-4.0, max=4.0)
            beta = beta - update

        corrected_log, corrected, moment, _ = distribution(beta)
        unresolved = entropy_active & (moment.abs() > moment_tolerance)
        if bool(unresolved.any().item()):
            radius = torch.ones_like(beta)
            lower = beta - radius
            upper = beta + radius
            for _ in range(16):
                *_, lower_moment, _ = distribution(lower)
                *_, upper_moment, _ = distribution(upper)
                bracketed = (lower_moment <= 0.0) & (upper_moment >= 0.0)
                expand = unresolved & ~bracketed
                radius = torch.where(expand, radius * 2.0, radius)
                lower = torch.where(expand, beta - radius, lower)
                upper = torch.where(expand, beta + radius, upper)
            *_, lower_moment, _ = distribution(lower)
            *_, upper_moment, _ = distribution(upper)
            if bool(
                (
                    unresolved
                    & ((lower_moment > 0.0) | (upper_moment < 0.0))
                )
                .any()
                .item()
            ):
                raise RuntimeError(
                    "self-information exponential failed to bracket moment"
                )
            for _ in range(64):
                midpoint = (lower + upper) / 2.0
                *_, midpoint_moment, _ = distribution(midpoint)
                lower = torch.where(
                    unresolved & (midpoint_moment < 0.0),
                    midpoint,
                    lower,
                )
                upper = torch.where(
                    unresolved & (midpoint_moment >= 0.0),
                    midpoint,
                    upper,
                )
            beta = torch.where(unresolved, (lower + upper) / 2.0, beta)
            corrected_log, corrected, moment, _ = distribution(beta)
        if bool(
            (entropy_active & (moment.abs() > moment_tolerance)).any().item()
        ):
            raise RuntimeError(
                "self-information exponential moment solve did not converge"
            )
        return corrected_log, corrected, beta, moment

    def retracted(scale: torch.Tensor):
        if retraction_mode == "self_information_mixture":
            mixture_ratio = 1.0 + scale.unsqueeze(-1) * consensus
            unnormalized = probability * mixture_ratio
            normalizer = unnormalized.sum(
                dim=-1,
                keepdim=True,
                dtype=torch.float64,
            ).to(dtype)
            target = unnormalized / normalizer
            target_log = (
                log_probability
                + mixture_ratio.log()
                - normalizer.log()
            )
            minimum_ratio = mixture_ratio.min(dim=-1).values
            information_multiplier = torch.zeros_like(scale)
            information_moment = weighted_sum(target, entropy_direction)
        elif retraction_mode == "self_information_exponential":
            (
                target_log,
                target,
                information_multiplier,
                information_moment,
            ) = corrected_exponential(scale)
            log_ratio = target_log - log_probability
            minimum_ratio = log_ratio.amin(dim=-1).exp().clamp_min(tiny)
        else:
            target_log = (
                log_probability + scale.unsqueeze(-1) * consensus
            ).log_softmax(dim=-1)
            target = target_log.exp()
            minimum_ratio = torch.ones_like(scale)
            information_multiplier = torch.zeros_like(scale)
            information_moment = weighted_sum(target, entropy_direction)
        forward_kl = weighted_sum(
            probability,
            log_probability - target_log,
        )
        reverse_kl = weighted_sum(
            target,
            target_log - log_probability,
        )
        return (
            target_log,
            target,
            forward_kl,
            reverse_kl,
            torch.maximum(forward_kl, reverse_kl),
            minimum_ratio,
            information_multiplier,
            information_moment,
        )

    (
        target_log,
        target,
        forward_kl,
        reverse_kl,
        worst_kl,
        minimum_ratio,
        information_multiplier,
        information_moment,
    ) = retracted(effective_step)
    needs_clip = worst_kl > float(max_target_kl)
    lower = torch.zeros_like(effective_step)
    upper = effective_step
    if bool(needs_clip.any().item()):
        for _ in range(32):
            midpoint = (lower + upper) / 2
            evaluated_scale = torch.where(
                needs_clip,
                midpoint,
                effective_step,
            )
            *_, midpoint_kl, _, _, _ = retracted(evaluated_scale)
            lower = torch.where(
                needs_clip & (midpoint_kl <= float(max_target_kl)),
                midpoint,
                lower,
            )
            upper = torch.where(
                needs_clip & (midpoint_kl > float(max_target_kl)),
                midpoint,
                upper,
            )
    effective_step = torch.where(needs_clip, lower, effective_step)
    (
        target_log,
        target,
        forward_kl,
        reverse_kl,
        worst_kl,
        minimum_ratio,
        information_multiplier,
        information_moment,
    ) = retracted(effective_step)
    base_cross_entropy_change = -weighted_sum(
        target,
        log_probability,
    ) + weighted_sum(probability, log_probability)
    target_entropy_change = (
        -weighted_sum(target, target_log)
        + weighted_sum(probability, log_probability)
    )
    target_values = torch.stack(
        [
            forward_kl,
            reverse_kl,
            worst_kl,
            effective_step,
            minimum_ratio,
            base_cross_entropy_change,
            target_entropy_change,
            information_multiplier,
            information_moment,
        ]
    )
    if not bool(torch.isfinite(target).all().item()) or not bool(
        torch.isfinite(target_values).all().item()
    ):
        raise RuntimeError("Fisher consensus target is nonfinite")
    kl_tolerance = max(1e-7, 128.0 * torch.finfo(dtype).eps)
    if bool(
        (worst_kl > float(max_target_kl) + kl_tolerance).any().item()
    ):
        raise RuntimeError("Fisher consensus target exceeds KL trust region")
    if preserves_self_information:
        information_tolerance = max(
            1e-12,
            128.0
            * torch.finfo(dtype).eps
            * (
                1.0
                + weighted_sum(probability, log_probability.abs()).max().item()
            ),
        )
        if bool((minimum_ratio <= 0.0).any().item()):
            raise RuntimeError("self-information retraction is not positive")
        if bool(
            (base_cross_entropy_change.abs() > information_tolerance)
            .any()
            .item()
        ):
            raise RuntimeError(
                "self-information retraction violates base cross entropy: "
                f"max residual={base_cross_entropy_change.abs().max().item():.6g}, "
                f"tolerance={information_tolerance:.6g}, "
                f"max moment={information_moment.abs().max().item():.6g}"
            )
        if bool(
            (target_entropy_change > information_tolerance).any().item()
        ):
            raise RuntimeError("self-information retraction increases entropy")
    consensus_energy = weighted_sum(probability, consensus.square())
    pair_cosine_to_mean = (
        weighted_sum(
            pair_probability,
            contrast * raw_mean.unsqueeze(0),
        )
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
            "positivity_limited": positivity_limited.to(dtype).detach(),
            "minimum_mixture_ratio": minimum_ratio.detach(),
            "self_information_multiplier": information_multiplier.detach(),
            "self_information_moment": information_moment.detach(),
            "base_cross_entropy_change": (
                base_cross_entropy_change.detach()
            ),
            "entropy_kl_identity_residual": (
                target_entropy_change
                + reverse_kl
                - base_cross_entropy_change
            ).detach(),
            "entropy_gradient_energy": entropy_energy.detach(),
            "entropy_alignment_before": entropy_alignment_before.detach(),
            "entropy_alignment_after": entropy_alignment_after.detach(),
            "first_order_entropy_change": (
                -entropy_alignment_after
            ).detach(),
            "retained_direction_energy_fraction": (
                retained_direction_energy_fraction.detach()
            ),
            "target_entropy_change": target_entropy_change.detach(),
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
    retraction_mode: str = "exponential",
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
        retraction_mode=retraction_mode,
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
