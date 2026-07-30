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


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> float:
    selected = value[mask]
    return float(selected.mean().detach()) if selected.numel() else 0.0


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
    metrics = result.metrics
    residual_energy = _masked_mean(metrics["residual_energy"], selected_mask)
    guide_energy = _masked_mean(metrics["guide_energy"], selected_mask)
    nuisance_energy = _masked_mean(metrics["nuisance_energy"], selected_mask)
    target_kl = _masked_mean(metrics["target_kl"], selected_mask)
    effective_step = _masked_mean(
        metrics["effective_step_size"], selected_mask
    )
    active_fraction = _masked_mean(
        metrics["active_projection"].to(torch.float32), selected_mask
    )
    collapsed_fraction = _masked_mean(
        (metrics["residual_energy"] <= self._finod_residual_energy_threshold).to(
            torch.float32
        ),
        selected_mask,
    )
    alignment_before = _masked_mean(
        metrics["guide_nuisance_alignment"], selected_mask
    )
    alignment_after = _masked_mean(
        metrics["residual_nuisance_alignment"], selected_mask
    )
    positive_alignment_after_fraction = _masked_mean(
        (metrics["residual_nuisance_alignment"] > 1e-5).to(torch.float32),
        selected_mask,
    )
    clipped_target_fraction = _masked_mean(
        (
            metrics["effective_step_size"]
            < float(self._finod_step_size) - 1e-7
        ).to(torch.float32),
        selected_mask,
    )
    if self.accelerator.is_main_process:
        print(
            json.dumps(
                {
                    "event": "finod_loss",
                    "loss": float(loss.detach().float()),
                    "selected_positions": int(rollout_positions.numel()),
                    "retained_tokens": int(selected_mask.sum()),
                    "guide_energy": guide_energy,
                    "nuisance_energy": nuisance_energy,
                    "residual_energy": residual_energy,
                    "retained_energy_fraction": residual_energy
                    / max(guide_energy, 1e-12),
                    "active_projection_fraction": active_fraction,
                    "collapsed_residual_fraction": collapsed_fraction,
                    "alignment_before": alignment_before,
                    "alignment_after": alignment_after,
                    "positive_alignment_after_fraction": (
                        positive_alignment_after_fraction
                    ),
                    "target_kl": target_kl,
                    "effective_step_size": effective_step,
                    "clipped_target_fraction": clipped_target_fraction,
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
