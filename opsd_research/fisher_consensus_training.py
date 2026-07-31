"""Memory-bounded training hook for answer-free Fisher consensus guidance."""

from __future__ import annotations

import json

import torch
from trl.trainer.utils import empty_cache

from .finod import select_rollout_positions
from .finod_training import _selected_logits, _teacher_context
from .fisher_consensus import fisher_consensus_loss


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
    local = torch.stack(
        [
            count,
            loss.detach().to(torch.float64) * count,
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

    return {
        "retained_tokens": int(global_count),
        "loss": average(1),
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
    )
    metrics = _aggregate_consensus_metrics(
        self,
        loss=loss,
        metrics=result.metrics,
        selected_mask=selected_mask,
        step_size=float(self._fisher_step_size),
        consensus_energy_threshold=float(
            self._fisher_consensus_energy_threshold
        ),
    )
    if self.accelerator.is_main_process:
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
    )
    empty_cache()

    if return_outputs:
        class MinimalOutput:
            def __init__(self, output_loss):
                self.loss = output_loss

        return loss, MinimalOutput(loss)
    return loss
