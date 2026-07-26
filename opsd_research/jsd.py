from __future__ import annotations

import torch


def exact_forward_kl_vocab_chunked(
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
    """Compute OPSD's full-vocabulary forward KL in bounded working memory.

    This is the beta=0 branch of the official generalized JSD implementation.
    Vocabulary chunks only change the reduction order; no tokens are dropped or
    renormalized, so this is not the upstream top-k approximation.
    """
    if beta != 0:
        raise ValueError("exact chunked OPSD loss currently requires beta=0")
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
        contributions = torch.exp(teacher_log_probs) * (
            teacher_log_probs - student_log_probs
        )
        if token_clip is not None:
            contributions = contributions.clamp(max=token_clip)
        if mask is not None:
            contributions = contributions * mask.unsqueeze(-1)
        total = total + contributions.sum()

    return total / denominator
