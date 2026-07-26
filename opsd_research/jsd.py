from __future__ import annotations

import torch


def exact_generalized_jsd_vocab_chunked(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor | None = None,
    *,
    beta: float = 0.0,
    temperature: float = 1.0,
    reduction: str = "batchmean",
    logits_are_probs: bool = False,
    top_k: int | None = None,
    token_clip: float | None = None,
    chunk_size: int,
) -> torch.Tensor:
    """Compute the upstream generalized-JSD loss in bounded working memory.

    ``beta=0`` is teacher-to-student forward KL; ``beta=1`` is reverse KL;
    and intermediate values use the upstream mixture
    ``m=(1-beta)p_student + beta p_teacher``.  Chunks only change reduction
    order: the loss remains full-vocabulary, with no top-k truncation or
    renormalisation.
    """
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be in [0, 1]")
    if logits_are_probs:
        raise ValueError("exact chunked OPSD loss requires logits, not probabilities")
    if top_k not in (None, 0):
        raise ValueError("exact chunked OPSD loss is incompatible with top-k loss")
    if reduction != "batchmean":
        raise ValueError("exact chunked OPSD loss currently requires batchmean reduction")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            "student and teacher logits must have identical shapes, got "
            f"{tuple(student_logits.shape)} and {tuple(teacher_logits.shape)}"
        )
    if student_logits.ndim != 3:
        raise ValueError("expected logits shaped [batch, sequence, vocabulary]")

    student_scaled = student_logits / temperature
    teacher_scaled = teacher_logits / temperature
    student_log_normalizer = torch.logsumexp(
        student_scaled, dim=-1, keepdim=True
    )
    teacher_log_normalizer = torch.logsumexp(
        teacher_scaled, dim=-1, keepdim=True
    )

    mask = None
    if labels is not None:
        if labels.shape != student_logits.shape[:2]:
            raise ValueError(
                "labels must match the batch and sequence dimensions of logits"
            )
        mask = labels != -100
        denominator = mask.sum()
        if int(denominator.item()) == 0:
            raise ValueError("cannot reduce an OPSD batch with no unmasked tokens")
    else:
        denominator = student_logits.new_tensor(student_logits.shape[0])

    total = student_logits.new_zeros(())
    vocabulary_size = student_logits.shape[-1]
    for start in range(0, vocabulary_size, chunk_size):
        stop = min(start + chunk_size, vocabulary_size)
        student_log_probs = (
            student_scaled[..., start:stop] - student_log_normalizer
        )
        teacher_log_probs = (
            teacher_scaled[..., start:stop] - teacher_log_normalizer
        )
        if beta == 0.0:
            contributions = torch.exp(teacher_log_probs) * (
                teacher_log_probs - student_log_probs
            )
        elif beta == 1.0:
            contributions = torch.exp(student_log_probs) * (
                student_log_probs - teacher_log_probs
            )
        else:
            beta_tensor = torch.as_tensor(
                beta, dtype=student_log_probs.dtype, device=student_log_probs.device
            )
            mixture_log_probs = torch.logaddexp(
                student_log_probs + torch.log1p(-beta_tensor),
                teacher_log_probs + torch.log(beta_tensor),
            )
            contributions = beta_tensor * torch.exp(teacher_log_probs) * (
                teacher_log_probs - mixture_log_probs
            ) + (1 - beta_tensor) * torch.exp(student_log_probs) * (
                student_log_probs - mixture_log_probs
            )
        if token_clip is not None:
            contributions = contributions.clamp(max=token_clip)
        if mask is not None:
            contributions = contributions * mask.unsqueeze(-1)
        total = total + contributions.sum()

    return total / denominator


def exact_forward_kl_vocab_chunked(*args, **kwargs) -> torch.Tensor:
    """Backward-compatible beta=0 entrypoint for the reproduced protocol."""
    if kwargs.get("beta", 0.0) != 0.0:
        raise ValueError("exact_forward_kl_vocab_chunked requires beta=0")
    return exact_generalized_jsd_vocab_chunked(*args, **kwargs)
