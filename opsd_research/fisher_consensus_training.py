"""Memory-bounded training hook for answer-free Fisher consensus guidance."""

from __future__ import annotations

import json

import torch
from trl.trainer.utils import empty_cache

from .finod import select_rollout_positions
from .finod_training import _selected_logits, _teacher_context
from .fisher_consensus import (
    FisherEdgeBoost,
    fisher_consensus_loss,
    fisher_edge_boost_weights,
    fisher_score_signature,
)


def _distributed_fisher_edge(
    self,
    signature: torch.Tensor,
    *,
    threshold: float,
) -> tuple[torch.Tensor, FisherEdgeBoost]:
    """All-gather detached signatures and return this rank's boost weights."""
    if signature.ndim != 2:
        raise RuntimeError("local Fisher signature must be a matrix")
    local_count = int(signature.shape[0])
    accelerator = self.accelerator
    gathered = accelerator.gather(signature.detach())
    process_count = int(getattr(accelerator, "num_processes", 1))
    expected = process_count * local_count
    if gathered.shape[0] != expected:
        raise RuntimeError(
            f"gathered {gathered.shape[0]} Fisher signatures, expected "
            f"{expected}"
        )
    result = fisher_edge_boost_weights(
        gathered,
        threshold=float(threshold),
    )
    process_index = int(getattr(accelerator, "process_index", 0))
    start = process_index * local_count
    stop = start + local_count
    return result.weights[start:stop].to(signature.device), result


def _sequence(
    inputs,
    *,
    name: str,
    generation_ids: torch.Tensor,
    generation_attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    prompts = inputs[f"{name}_prompts"]
    prompt_mask = inputs[f"{name}_prompt_attention_mask"]
    width = int(inputs[f"{name}_prompt_length"])
    if prompts.shape[1] != width or prompt_mask.shape != prompts.shape:
        raise RuntimeError(f"{name} prompt tensors are inconsistent")
    return (
        torch.cat([prompts, generation_ids], dim=1),
        torch.cat([prompt_mask, generation_attention_mask], dim=1),
        width,
    )


def _aggregate_consensus_metrics(
    self,
    *,
    loss: torch.Tensor,
    metrics: dict[str, torch.Tensor],
    selected_mask: torch.Tensor,
    step_size: float,
    consensus_energy_threshold: float,
    example_weights: torch.Tensor | None = None,
) -> dict[str, float | int]:
    """Reduce token-weighted consensus diagnostics across data-parallel ranks."""
    mask = selected_mask.to(torch.bool)
    count = mask.sum().to(torch.float64)

    def total(name: str) -> torch.Tensor:
        return metrics[name][mask].detach().to(torch.float64).sum()

    energy = metrics["consensus_energy"][mask].detach().to(torch.float64)
    effective = (
        metrics["effective_step_size"][mask].detach().to(torch.float64)
    )
    if example_weights is None:
        example_weights = torch.ones(
            mask.shape[0],
            dtype=torch.float64,
            device=mask.device,
        )
    if example_weights.shape != (mask.shape[0],):
        raise RuntimeError("Fisher boost weights must match local examples")
    token_weights = example_weights.to(torch.float64).unsqueeze(1).expand(
        mask.shape
    )
    boosted_optimization_total = (
        metrics["optimization_per_token"].detach().to(torch.float64)
        * token_weights
    )[mask].sum()
    local = torch.stack(
        [
            count,
            total("target_student_kl"),
            total("agreement"),
            total("mean_direction_energy"),
            total("mean_pair_energy"),
            energy.sum(),
            total("mean_pair_cosine"),
            total("min_pair_cosine"),
            total("target_forward_kl"),
            total("target_reverse_kl"),
            total("target_kl"),
            effective.sum(),
            (
                energy <= float(consensus_energy_threshold)
            ).to(torch.float64).sum(),
            (effective < float(step_size) - 1e-7).to(torch.float64).sum(),
            total("student_anchor_forward_kl"),
            total("student_anchor_reverse_kl"),
            total("target_student_kl"),
            total("fisher_alignment_gain"),
            total("fisher_alignment_cosine_proxy"),
            total("optimization_per_token"),
            boosted_optimization_total,
            total("entropy_gradient_energy"),
            total("entropy_alignment_before"),
            total("entropy_alignment_after"),
            total("first_order_entropy_change"),
            total("retained_direction_energy_fraction"),
            total("target_entropy_change"),
            total("positivity_limited"),
            total("minimum_mixture_ratio"),
            metrics["base_cross_entropy_change"][mask]
            .detach()
            .to(torch.float64)
            .abs()
            .sum(),
            metrics["entropy_kl_identity_residual"][mask]
            .detach()
            .to(torch.float64)
            .abs()
            .sum(),
        ]
    )
    local_max = metrics["target_kl"][mask].detach().to(torch.float64).max()
    accelerator = self.accelerator
    if hasattr(accelerator, "reduce"):
        reduced = accelerator.reduce(local, reduction="sum")
        max_kl = accelerator.gather(local_max.reshape(1)).max()
    else:
        reduced = local
        max_kl = local_max
    values = reduced.detach().cpu().tolist()
    global_count = values[0]
    if global_count <= 0:
        raise RuntimeError("Fisher consensus selected no valid tokens")

    def average(index: int) -> float:
        return values[index] / global_count

    student_target_loss = average(1)
    anchor_target_loss = average(9)
    return {
        "retained_tokens": int(global_count),
        "loss": student_target_loss,
        "agreement": average(2),
        "mean_direction_energy": average(3),
        "mean_pair_energy": average(4),
        "consensus_energy": average(5),
        "mean_pair_cosine": average(6),
        "min_pair_cosine": average(7),
        "target_forward_kl": average(8),
        "target_reverse_kl": average(9),
        "target_kl": average(10),
        "effective_step_size": average(11),
        "collapsed_consensus_fraction": average(12),
        "clipped_target_fraction": average(13),
        "max_observed_target_kl": float(max_kl.detach().cpu()),
        "anchor_target_loss": anchor_target_loss,
        "relative_loss_to_anchor": student_target_loss
        / max(anchor_target_loss, 1e-12),
        "improvement_over_anchor": (
            anchor_target_loss - student_target_loss
        ),
        "student_anchor_forward_kl": average(14),
        "student_anchor_reverse_kl": average(15),
        "target_student_kl": average(16),
        "fisher_alignment_gain": average(17),
        "fisher_alignment_cosine_proxy": average(18),
        "optimization_loss": average(19),
        "boosted_optimization_loss": average(20),
        "entropy_gradient_energy": average(21),
        "entropy_alignment_before": average(22),
        "entropy_alignment_after": average(23),
        "first_order_entropy_change": average(24),
        "retained_direction_energy_fraction": average(25),
        "target_entropy_change": average(26),
        "positivity_limited_fraction": average(27),
        "mean_minimum_mixture_ratio": average(28),
        "mean_absolute_base_cross_entropy_change": average(29),
        "mean_absolute_entropy_kl_identity_residual": average(30),
    }


def compute_loss_with_fisher_consensus(
    self,
    model,
    inputs,
    return_outputs: bool = False,
    num_items_in_batch=None,
):
    """Fit an early-prefix target formed by answer-free plan agreement."""
    student_width = int(inputs["student_prompt_length"])
    generation_ids = inputs["student_input_ids"][:, student_width:]
    generation_mask = inputs["student_attention_mask"][:, student_width:]
    labels = inputs["labels"][:, student_width:]
    valid_mask = labels != -100
    rollout_positions = select_rollout_positions(
        valid_mask,
        max_positions=int(self._fisher_positions_per_rollout),
        prefix_tokens=int(self._fisher_position_prefix_tokens),
    )
    selected_mask = valid_mask.index_select(1, rollout_positions)
    student_logits = _selected_logits(
        model,
        input_ids=inputs["student_input_ids"],
        attention_mask=inputs["student_attention_mask"],
        prompt_width=student_width,
        rollout_positions=rollout_positions,
    )

    base_ids, base_mask, base_width = _sequence(
        inputs,
        name="fisher_base",
        generation_ids=generation_ids,
        generation_attention_mask=generation_mask,
    )
    with torch.no_grad(), _teacher_context(self, model):
        base_logits = _selected_logits(
            model,
            input_ids=base_ids,
            attention_mask=base_mask,
            prompt_width=base_width,
            rollout_positions=rollout_positions,
        )
    empty_cache()

    guide_logits: list[torch.Tensor] = []
    control_logits: list[torch.Tensor] = []
    pair_count = int(inputs["fisher_pair_count"])
    if pair_count < 2:
        raise RuntimeError("Fisher consensus requires at least two pairs")
    for pair in range(pair_count):
        for kind, destination in (
            ("guide", guide_logits),
            ("control", control_logits),
        ):
            ids, mask, width = _sequence(
                inputs,
                name=f"fisher_{kind}_{pair}",
                generation_ids=generation_ids,
                generation_attention_mask=generation_mask,
            )
            with torch.no_grad(), _teacher_context(self, model):
                logits = _selected_logits(
                    model,
                    input_ids=ids,
                    attention_mask=mask,
                    prompt_width=width,
                    rollout_positions=rollout_positions,
                )
            destination.append(logits)
            del ids, mask
            empty_cache()

    stacked_guides = torch.stack(guide_logits, dim=0)
    stacked_controls = torch.stack(control_logits, dim=0)
    loss, result = fisher_consensus_loss(
        student_logits=student_logits,
        base_logits=base_logits,
        guide_logits=stacked_guides,
        control_logits=stacked_controls,
        token_mask=selected_mask,
        step_size=float(self._fisher_step_size),
        max_target_kl=float(self._fisher_max_target_kl),
        temperature=float(self.temperature),
        anchor_kl_weight=float(self._fisher_anchor_kl_weight),
        direction_mode=str(self._fisher_direction_mode),
        retraction_mode=str(self._fisher_retraction_mode),
    )
    boost_weights = torch.ones(
        student_logits.shape[0],
        dtype=torch.float32,
        device=student_logits.device,
    )
    edge_result = None
    signature_norms = None
    if bool(self._fisher_cross_problem_edge):
        signature, signature_norms = fisher_score_signature(
            base_logits=base_logits,
            consensus_direction=result.consensus_direction,
            token_mask=selected_mask,
            temperature=float(self.temperature),
        )
        boost_weights, edge_result = _distributed_fisher_edge(
            self,
            signature,
            threshold=float(self._fisher_edge_threshold),
        )
        loss = loss * boost_weights.mean().to(loss.dtype)
    metrics = _aggregate_consensus_metrics(
        self,
        loss=loss,
        metrics=result.metrics,
        selected_mask=selected_mask,
        step_size=float(self._fisher_step_size),
        consensus_energy_threshold=float(
            self._fisher_consensus_energy_threshold
        ),
        example_weights=boost_weights,
    )
    if self.accelerator.is_main_process:
        edge_metrics = {}
        if edge_result is not None:
            edge_metrics = {
                "fisher_edge_signed_mean": float(
                    edge_result.signed_edges.mean().detach().cpu()
                ),
                "fisher_edge_signed_min": float(
                    edge_result.signed_edges.min().detach().cpu()
                ),
                "fisher_edge_signed_max": float(
                    edge_result.signed_edges.max().detach().cpu()
                ),
                "fisher_edge_active_fraction": float(
                    edge_result.active_fraction.detach().cpu()
                ),
                "fisher_edge_mean_positive": float(
                    edge_result.mean_positive_edge.detach().cpu()
                ),
                "fisher_edge_max_positive": float(
                    edge_result.max_positive_edge.detach().cpu()
                ),
                "fisher_edge_boost_weight_mean": float(
                    edge_result.weights.mean().detach().cpu()
                ),
                "fisher_signature_norm_mean": float(
                    signature_norms.mean().detach().cpu()
                ),
            }
        print(
            json.dumps(
                {
                    "event": "fisher_consensus_loss",
                    "loss": metrics["loss"],
                    "pair_count": pair_count,
                    "selected_positions": int(rollout_positions.numel()),
                    "position_prefix_tokens": int(
                        self._fisher_position_prefix_tokens
                    ),
                    "max_target_kl": float(self._fisher_max_target_kl),
                    "anchor_kl_weight": float(
                        self._fisher_anchor_kl_weight
                    ),
                    "cross_problem_edge": bool(
                        self._fisher_cross_problem_edge
                    ),
                    "direction_mode": str(self._fisher_direction_mode),
                    "retraction_mode": str(self._fisher_retraction_mode),
                    **edge_metrics,
                    **metrics,
                },
                separators=(",", ":"),
            ),
            flush=True,
        )

    del (
        student_logits,
        base_logits,
        stacked_guides,
        stacked_controls,
        guide_logits,
        control_logits,
        result,
        base_ids,
        base_mask,
        boost_weights,
    )
    empty_cache()

    if return_outputs:
        class MinimalOutput:
            def __init__(self, output_loss):
                self.loss = output_loss

        return loss, MinimalOutput(loss)
    return loss
