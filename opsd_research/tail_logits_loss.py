from __future__ import annotations

from contextlib import nullcontext
import json
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
        divergence = getattr(self, "_opsd_token_divergence", None)
        if divergence is not None:
            from .jsd import canonical_negative_tolerance

            loss_tolerance = canonical_negative_tolerance(loss.dtype)
        if divergence is not None and (
            not bool(torch.isfinite(loss).item())
            or float(loss.detach()) < -loss_tolerance
        ):
            raise FloatingPointError(
                f"canonical {divergence} produced invalid loss {float(loss.detach())}"
            )
        if self.accelerator.is_main_process and divergence is not None:
            print(
                json.dumps(
                    {
                        "event": "canonical_divergence_loss",
                        "objective": divergence,
                        "value": float(loss.detach().float()),
                        "roundoff_tolerance": loss_tolerance,
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
            calls = int(getattr(self, "_opsd_divergence_loss_calls", 0)) + 1
            self._opsd_divergence_loss_calls = calls
            interval = int(self._opsd_divergence_diagnostics_interval)
            if calls % interval == 0:
                from .jsd import divergence_statistics_vocab_chunked

                diagnostics = divergence_statistics_vocab_chunked(
                    student_logits_for_loss,
                    teacher_logits_for_loss,
                    shifted_labels,
                    temperature=self.temperature,
                    chunk_size=int(self._opsd_vocab_chunk_size),
                )
                for name in ("forward_kl", "reverse_kl", "js"):
                    values = diagnostics[name]
                    if (
                        values["nonfinite_count"] > 0
                        or values["material_negative_count"] > 0
                    ):
                        raise FloatingPointError(
                            f"canonical divergence diagnostics failed for {name}: {values}"
                        )
                print(
                    json.dumps(
                        {
                            "event": "divergence_diagnostics",
                            "objective": divergence,
                            **diagnostics,
                        },
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
        del student_logits_for_loss, teacher_logits_for_loss

    # Trainer's normal progress display rounds this scalar to four decimal
    # places.  Preserve the unrounded value separately so a tiny forward KL
    # cannot be mistaken for a zero loss during a long-running experiment.
    if (
        self.accelerator.is_main_process
        and getattr(self, "_opsd_token_divergence", None) is None
    ):
        print(
            '{"event":"exact_forward_kl_loss",'
            f'"value":{float(loss.detach().float()):.10g}' + "}",
            flush=True,
        )

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
    aggregate = _aggregate_graf_metrics(
        self,
        base_loss=base_loss,
        branch_loss=branch_loss,
        total_loss=total,
        metrics=metrics,
    )
    base_value = aggregate["base_loss"]
    branch_value = aggregate["branch_loss"]
    total_value = aggregate["total_loss"]
    branch_to_base_ratio = abs(branch_value) / max(
        abs(base_value), 1e-12
    )
    self._graf_last_branch_metrics = aggregate
    if self.accelerator.is_main_process:
        print(
            '{"event":"graf_branch_loss",'
            '"metric_scope":"global_batch",'
            f'"active_forks":{aggregate["active_forks"]},'
            f'"effective_fork_weight":{aggregate["effective_fork_weight"]:.8f},'
            f'"base_loss":{base_value:.8f},'
            f'"branch_kl":{aggregate["branch_kl"]:.8f},'
            f'"entropy_floor":{aggregate["entropy_floor"]:.8f},'
            f'"weighted_loss":{branch_value:.8f},'
            f'"total_loss":{total_value:.8f},'
            f'"branch_to_base_ratio":{branch_to_base_ratio:.8f}' + "}",
            flush=True,
        )
    if return_outputs:
        outputs.loss = total
        return total, outputs
    return total


def _aggregate_graf_metrics(
    self,
    *,
    base_loss,
    branch_loss,
    total_loss,
    metrics: dict[str, float],
) -> dict[str, float]:
    """Reduce detached telemetry across ranks without changing optimization."""
    import torch

    evidence_weight = float(
        metrics.get("effective_fork_weight", metrics["active_forks"])
    )
    local = torch.tensor(
        [
            float(base_loss.detach()),
            float(branch_loss.detach()),
            float(total_loss.detach()),
            float(metrics["active_forks"]),
            evidence_weight,
            float(metrics["branch_kl"]) * evidence_weight,
            float(metrics["entropy_floor"]) * evidence_weight,
        ],
        dtype=torch.float64,
        device=base_loss.device,
    )
    accelerator = self.accelerator
    if hasattr(accelerator, "reduce"):
        reduced = accelerator.reduce(local, reduction="sum")
        world_size = int(getattr(accelerator, "num_processes", 1))
    else:
        reduced = local
        world_size = 1
    if world_size < 1:
        raise RuntimeError("accelerator num_processes must be positive")
    values = reduced.detach().cpu().tolist()
    total_weight = values[4]
    return {
        "base_loss": values[0] / world_size,
        "branch_loss": values[1] / world_size,
        "total_loss": values[2] / world_size,
        "active_forks": values[3],
        "effective_fork_weight": total_weight,
        "branch_kl": values[5] / total_weight if total_weight else 0.0,
        "entropy_floor": values[6] / total_weight if total_weight else 0.0,
    }
