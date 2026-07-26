from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
import torch.nn.functional as F
from accelerate.utils import is_peft_model
from trl.trainer.utils import empty_cache


def _generation_logits(outputs: Any, generation_length: int) -> torch.Tensor:
    expected = generation_length + 1
    if outputs.logits.shape[1] != expected:
        raise RuntimeError(
            "tail-logit forward returned an unexpected sequence dimension: "
            f"expected {expected}, got {outputs.logits.shape[1]}"
        )
    return outputs.logits[:, :-1, :]


def compute_loss_with_tail_logits(
    self,
    model,
    inputs,
    return_outputs: bool = False,
    num_items_in_batch=None,
):
    """Run the official OPSD loss without materializing prompt-token logits."""
    student_prompt_len = inputs["student_prompt_length"]
    sampled_token_ids = inputs["student_input_ids"][:, student_prompt_len:]
    shifted_labels = inputs["labels"][:, student_prompt_len:]
    generation_length = sampled_token_ids.shape[1]
    logits_to_keep = generation_length + 1

    outputs_student = model(
        input_ids=inputs["student_input_ids"],
        attention_mask=inputs["student_attention_mask"],
        logits_to_keep=logits_to_keep,
    )
    student_logits = _generation_logits(outputs_student, generation_length)

    if self.use_thinking_machines_loss:
        student_log_probs = F.log_softmax(
            student_logits / self.temperature, dim=-1
        )
        student_log_probs_sampled = torch.gather(
            student_log_probs, dim=-1, index=sampled_token_ids.unsqueeze(-1)
        ).squeeze(-1)
        del student_logits, student_log_probs
    else:
        student_logits_for_loss = student_logits
        del student_logits

    if return_outputs:
        class MinimalOutput:
            def __init__(self):
                self.loss = None

        minimal_output = MinimalOutput()

    del outputs_student
    empty_cache()

    if self.use_ema_teacher:
        adapter_context = self._ema_teacher_context(model)
    elif self.fixed_teacher and is_peft_model(model):
        adapter_context = self.accelerator.unwrap_model(model).disable_adapter()
    else:
        adapter_context = nullcontext()

    with torch.no_grad(), adapter_context:
        outputs_teacher = model(
            input_ids=inputs["teacher_input_ids"],
            attention_mask=inputs["teacher_attention_mask"],
            logits_to_keep=logits_to_keep,
        )
        teacher_logits = _generation_logits(outputs_teacher, generation_length)

        if self.use_thinking_machines_loss:
            teacher_log_probs = F.log_softmax(
                teacher_logits / self.temperature, dim=-1
            )
            teacher_log_probs_sampled = torch.gather(
                teacher_log_probs,
                dim=-1,
                index=sampled_token_ids.unsqueeze(-1),
            ).squeeze(-1)
            del teacher_logits, teacher_log_probs
        else:
            teacher_logits_for_loss = teacher_logits
            del teacher_logits

        del outputs_teacher
        empty_cache()

    if self.use_thinking_machines_loss:
        advantage = (
            teacher_log_probs_sampled - student_log_probs_sampled
        ).detach()
        if shifted_labels is not None:
            mask = shifted_labels != -100
            advantage = advantage[mask]
            student_log_probs_sampled_masked = student_log_probs_sampled[mask]
        else:
            student_log_probs_sampled_masked = student_log_probs_sampled

        loss = -(advantage * student_log_probs_sampled_masked).mean()
        del (
            student_log_probs_sampled,
            teacher_log_probs_sampled,
            advantage,
            student_log_probs_sampled_masked,
        )
    else:
        loss = self.generalized_jsd_loss(
            student_logits=student_logits_for_loss,
            teacher_logits=teacher_logits_for_loss,
            labels=shifted_labels,
            beta=self.beta,
            temperature=self.temperature,
            top_k=self.top_k_loss,
            token_clip=self.jsd_token_clip,
        )
        del student_logits_for_loss, teacher_logits_for_loss

    empty_cache()
    if return_outputs:
        minimal_output.loss = loss
        return loss, minimal_output
    return loss


def compute_loss_with_graf_routing(
    self,
    model,
    inputs,
    return_outputs: bool = False,
    num_items_in_batch=None,
):
    """Add GRAF's frozen branch-target objective to exact OPSD distillation.

    The normal tail-only OPSD path runs first and releases its logits.  The
    routing term then performs a compact, action-only policy forward.  This
    ordering keeps the 4B 8x40GB recipe within the same memory envelope.
    """
    result = compute_loss_with_tail_logits(
        self,
        model,
        inputs,
        return_outputs=return_outputs,
        num_items_in_batch=num_items_in_batch,
    )
    if return_outputs:
        base_loss, outputs = result
    else:
        base_loss = result
        outputs = None
    from .graf_branch_objective import graf_branch_loss

    branch_loss, metrics = graf_branch_loss(
        model=model,
        inputs=inputs,
        tokenizer=self.processing_class,
        routing_targets=self._graf_routing_targets,
        branch_loss_weight=self._graf_branch_loss_weight,
        entropy_floor_fraction=self._graf_entropy_floor_fraction,
    )
    total = base_loss + branch_loss
    self._graf_last_branch_metrics = metrics
    if self.accelerator.is_main_process:
        print(
            '{"event":"graf_branch_loss",'
            f'"active_forks":{metrics["active_forks"]},'
            f'"effective_fork_weight":{metrics.get("effective_fork_weight", metrics["active_forks"]):.8f},'
            f'"branch_kl":{metrics["branch_kl"]:.8f},'
            f'"entropy_floor":{metrics["entropy_floor"]:.8f},'
            f'"weighted_loss":{float(branch_loss.detach()):.8f}' + "}",
            flush=True,
        )
    if return_outputs:
        outputs.loss = total
        return total, outputs
    return total
