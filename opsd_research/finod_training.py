"""Memory-bounded on-policy FiNOD training hook."""

from __future__ import annotations

from contextlib import nullcontext
import json

import torch
from accelerate.utils import is_peft_model
from trl.trainer.utils import empty_cache

from .finod import fisher_projected_loss, select_rollout_positions


def _selected_logits(
    model,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_width: int,
    rollout_positions: torch.Tensor,
) -> torch.Tensor:
    prediction_positions = prompt_width - 1 + rollout_positions
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        logits_to_keep=prediction_positions,
    )
    logits = outputs.logits
    expected = (
        input_ids.shape[0],
        rollout_positions.numel(),
        logits.shape[-1],
    )
    if logits.shape != expected:
        raise RuntimeError(
            "FiNOD sparse-logit forward returned shape "
            f"{tuple(logits.shape)}, expected {expected}"
        )
    del outputs
    return logits


def _control_sequence(
    inputs,
    *,
    name: str,
    generation_ids: torch.Tensor,
    generation_attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    prompts = inputs[f"finod_{name}_prompts"]
    prompt_mask = inputs[f"finod_{name}_prompt_attention_mask"]
    width = int(inputs[f"finod_{name}_prompt_length"])
    if prompts.shape[1] != width or prompt_mask.shape != prompts.shape:
        raise RuntimeError(f"FiNOD {name} prompt tensors are inconsistent")
    return (
        torch.cat([prompts, generation_ids], dim=1),
        torch.cat([prompt_mask, generation_attention_mask], dim=1),
        width,
    )


def _teacher_context(self, model):
    if not self.fixed_teacher:
        raise RuntimeError("FiNOD requires a fixed teacher")
    if not is_peft_model(model):
        raise RuntimeError("FiNOD fixed teacher requires a PEFT student")
    return self.accelerator.unwrap_model(model).disable_adapter()


def _aggregate_finod_metrics(
    self,
    *,
    loss: torch.Tensor,
    metrics: dict[str, torch.Tensor],
    selected_mask: torch.Tensor,
    step_size: float,
    residual_energy_threshold: float,
) -> dict[str, float | int]:
    """Reduce token-weighted FiNOD diagnostics across every data-parallel rank."""
    mask = selected_mask.to(torch.bool)
    count = mask.sum().to(torch.float64)

    def total(name: str) -> torch.Tensor:
        return metrics[name][mask].detach().to(torch.float64).sum()

    active = metrics["active_projection"][mask].detach().to(torch.float64)
    residual = metrics["residual_energy"][mask].detach().to(torch.float64)
    alignment_after = (
        metrics["residual_nuisance_alignment"][mask].detach().to(torch.float64)
    )
    effective_step = (
        metrics["effective_step_size"][mask].detach().to(torch.float64)
    )
    local = torch.stack(
        [
            count,
            loss.detach().to(torch.float64) * count,
            total("guide_energy"),
            total("nuisance_energy"),
            residual.sum(),
            total("target_kl"),
            effective_step.sum(),
            active.sum(),
            (residual <= residual_energy_threshold).to(torch.float64).sum(),
            total("guide_nuisance_alignment"),
            alignment_after.sum(),
            (alignment_after > 1e-5).to(torch.float64).sum(),
            (effective_step < step_size - 1e-7).to(torch.float64).sum(),
        ]
    )
    local_max_kl = metrics["target_kl"][mask].detach().to(torch.float64).max()
    accelerator = self.accelerator
    if hasattr(accelerator, "reduce"):
        reduced = accelerator.reduce(local, reduction="sum")
        gathered_max_kl = accelerator.gather(local_max_kl.reshape(1))
        max_kl = gathered_max_kl.max()
    else:
        reduced = local
        max_kl = local_max_kl
    values = reduced.detach().cpu().tolist()
    global_count = values[0]
    if global_count <= 0:
        raise RuntimeError("FiNOD selected no valid rollout tokens")

    def average(index: int) -> float:
        return values[index] / global_count

    return {
        "retained_tokens": int(global_count),
        "loss": average(1),
        "guide_energy": average(2),
        "nuisance_energy": average(3),
        "residual_energy": average(4),
        "target_kl": average(5),
        "effective_step_size": average(6),
        "active_projection_fraction": average(7),
        "collapsed_residual_fraction": average(8),
        "alignment_before": average(9),
        "alignment_after": average(10),
        "positive_alignment_after_fraction": average(11),
        "clipped_target_fraction": average(12),
        "max_observed_target_kl": float(max_kl.detach().cpu()),
    }


def compute_loss_with_finod(
    self,
    model,
    inputs,
    return_outputs: bool = False,
    num_items_in_batch=None,
):
    """Fit a clipped target after removing answer-control Fisher alignment."""
    student_prompt_width = int(inputs["student_prompt_length"])
    generation_ids = inputs["student_input_ids"][:, student_prompt_width:]
    generation_attention_mask = inputs["student_attention_mask"][
        :, student_prompt_width:
    ]
    labels = inputs["labels"][:, student_prompt_width:]
    valid_mask = labels != -100
    rollout_positions = select_rollout_positions(
        valid_mask,
        max_positions=int(self._finod_positions_per_rollout),
    )
    selected_mask = valid_mask.index_select(1, rollout_positions)

    student_logits = _selected_logits(
        model,
        input_ids=inputs["student_input_ids"],
        attention_mask=inputs["student_attention_mask"],
        prompt_width=student_prompt_width,
        rollout_positions=rollout_positions,
    )

    guide_width = int(inputs["teacher_prompt_length"])
    base_ids, base_mask, base_width = _control_sequence(
        inputs,
        name="base",
        generation_ids=generation_ids,
        generation_attention_mask=generation_attention_mask,
    )
    answer_ids, answer_mask, answer_width = _control_sequence(
        inputs,
        name="answer",
        generation_ids=generation_ids,
        generation_attention_mask=generation_attention_mask,
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
    with torch.no_grad(), _teacher_context(self, model):
        guide_logits = _selected_logits(
            model,
            input_ids=inputs["teacher_input_ids"],
            attention_mask=inputs["teacher_attention_mask"],
            prompt_width=guide_width,
            rollout_positions=rollout_positions,
        )
    empty_cache()
    with torch.no_grad(), _teacher_context(self, model):
        answer_logits = _selected_logits(
            model,
            input_ids=answer_ids,
            attention_mask=answer_mask,
            prompt_width=answer_width,
            rollout_positions=rollout_positions,
        )
    empty_cache()

    loss, result = fisher_projected_loss(
        student_logits=student_logits,
        guide_logits=guide_logits,
        base_logits=base_logits,
        nuisance_logits=answer_logits,
        token_mask=selected_mask,
        step_size=float(self._finod_step_size),
        max_target_kl=float(self._finod_max_target_kl),
        nuisance_strength_threshold=float(
            self._finod_nuisance_strength_threshold
        ),
        temperature=float(self.temperature),
    )
    metrics = _aggregate_finod_metrics(
        self,
        loss=loss,
        metrics=result.metrics,
        selected_mask=selected_mask,
        step_size=float(self._finod_step_size),
        residual_energy_threshold=float(
            self._finod_residual_energy_threshold
        ),
    )
    if self.accelerator.is_main_process:
        print(
            json.dumps(
                {
                    "event": "finod_loss",
                    "loss": metrics["loss"],
                    "selected_positions": int(rollout_positions.numel()),
                    **metrics,
                    "retained_energy_fraction": metrics["residual_energy"]
                    / max(metrics["guide_energy"], 1e-12),
                    "max_target_kl": float(self._finod_max_target_kl),
                },
                separators=(",", ":"),
            ),
            flush=True,
        )

    del (
        student_logits,
        base_logits,
        guide_logits,
        answer_logits,
        result,
        base_ids,
        base_mask,
        answer_ids,
        answer_mask,
    )
    empty_cache()

    if return_outputs:
        class MinimalOutput:
            def __init__(self, output_loss):
                self.loss = output_loss

        return loss, MinimalOutput(loss)
    return loss
