from __future__ import annotations

import torch


def canonical_negative_tolerance(dtype: torch.dtype) -> float:
    """Return the material-negativity threshold for a canonical divergence.

    A vocabulary KL is a signed reduction whose exact result is non-negative.
    Near equality, positive and negative vocabulary contributions cancel and
    the FP32 reduction can retain a few ulps of negative residue.  Sixty-four
    working-precision epsilons is still below 8e-6 nats in FP32 while avoiding
    false failures seen with 151k-way vocabularies.  Raw negatives remain
    visible in telemetry; this threshold only classifies whether they are
    material.
    """
    if dtype not in {torch.float32, torch.float64}:
        raise ValueError("canonical divergence tolerance requires a work dtype")
    return 64.0 * float(torch.finfo(dtype).eps)


@torch.no_grad()
def divergence_statistics_vocab_chunked(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor | None = None,
    *,
    temperature: float = 1.0,
    chunk_size: int,
) -> dict[str, object]:
    """Summarize canonical per-token divergences without retaining vocabulary tensors."""
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
    if labels is not None and labels.shape != student_logits.shape[:2]:
        raise ValueError("labels must match the batch and sequence dimensions of logits")

    work_dtype = torch.float64 if student_logits.dtype == torch.float64 else torch.float32
    student_scaled = student_logits.to(work_dtype) / temperature
    teacher_scaled = teacher_logits.to(work_dtype) / temperature
    student_log_normalizer = torch.logsumexp(student_scaled, dim=-1, keepdim=True)
    teacher_log_normalizer = torch.logsumexp(teacher_scaled, dim=-1, keepdim=True)
    shape = student_logits.shape[:2]
    per_token = {
        "forward_kl": student_scaled.new_zeros(shape),
        "reverse_kl": student_scaled.new_zeros(shape),
        "js": student_scaled.new_zeros(shape),
        "student_entropy": student_scaled.new_zeros(shape),
        "teacher_entropy": student_scaled.new_zeros(shape),
    }
    vocabulary_size = student_logits.shape[-1]
    for start in range(0, vocabulary_size, chunk_size):
        stop = min(start + chunk_size, vocabulary_size)
        student_log_probs = student_scaled[..., start:stop] - student_log_normalizer
        teacher_log_probs = teacher_scaled[..., start:stop] - teacher_log_normalizer
        student_probs = torch.exp(student_log_probs)
        teacher_probs = torch.exp(teacher_log_probs)
        mixture_log_probs = torch.logaddexp(
            student_log_probs, teacher_log_probs
        ) - student_log_probs.new_tensor(2.0).log()
        per_token["forward_kl"] += (
            teacher_probs * (teacher_log_probs - student_log_probs)
        ).sum(dim=-1)
        per_token["reverse_kl"] += (
            student_probs * (student_log_probs - teacher_log_probs)
        ).sum(dim=-1)
        per_token["js"] += 0.5 * (
            teacher_probs * (teacher_log_probs - mixture_log_probs)
            + student_probs * (student_log_probs - mixture_log_probs)
        ).sum(dim=-1)
        per_token["student_entropy"] -= (student_probs * student_log_probs).sum(dim=-1)
        per_token["teacher_entropy"] -= (teacher_probs * teacher_log_probs).sum(dim=-1)

    mask = torch.ones(shape, dtype=torch.bool, device=student_logits.device)
    if labels is not None:
        mask = labels != -100
    token_count = int(mask.sum().item())
    if token_count == 0:
        raise ValueError("cannot summarize an OPSD batch with no unmasked tokens")

    def summarize(values: torch.Tensor) -> dict[str, float | int]:
        selected = values[mask]
        finite_mask = torch.isfinite(selected)
        finite = selected[finite_mask]
        nonfinite_count = int((~finite_mask).sum().item())
        tolerance = canonical_negative_tolerance(work_dtype)
        if finite.numel() == 0:
            return {
                "mean": float("nan"),
                "min": float("nan"),
                "p50": float("nan"),
                "p90": float("nan"),
                "p99": float("nan"),
                "max": float("nan"),
                "nonfinite_count": nonfinite_count,
                "negative_count": 0,
                "roundoff_negative_count": 0,
                "material_negative_count": 0,
                "roundoff_tolerance": tolerance,
            }
        quantiles = torch.quantile(
            finite, finite.new_tensor([0.50, 0.90, 0.99])
        )
        negative = finite < 0.0
        material_negative = finite < -tolerance
        return {
            "mean": float(finite.mean().item()),
            "min": float(finite.min().item()),
            "p50": float(quantiles[0].item()),
            "p90": float(quantiles[1].item()),
            "p99": float(quantiles[2].item()),
            "max": float(finite.max().item()),
            "nonfinite_count": nonfinite_count,
            "negative_count": int(negative.sum().item()),
            "roundoff_negative_count": int(
                (negative & ~material_negative).sum().item()
            ),
            "material_negative_count": int(material_negative.sum().item()),
            "roundoff_tolerance": tolerance,
        }

    valid_rank = mask.to(torch.int64).cumsum(dim=1) - 1
    valid_length = mask.sum(dim=1, keepdim=True).clamp_min(1)
    normalized_bin = (
        4 * valid_rank // valid_length
    ).clamp(min=0, max=3)
    position_quartiles = []
    for bin_index in range(4):
        bin_mask = mask & (normalized_bin == bin_index)
        bin_count = int(bin_mask.sum().item())
        row: dict[str, float | int | None] = {
            "start_fraction": bin_index / 4,
            "end_fraction": (bin_index + 1) / 4,
            "token_count": bin_count,
        }
        for name, values in per_token.items():
            selected = values[bin_mask]
            finite = selected[torch.isfinite(selected)]
            row[f"{name}_mean"] = (
                float(finite.mean().item()) if finite.numel() else None
            )
        position_quartiles.append(row)

    return {
        "token_count": token_count,
        **{name: summarize(values) for name, values in per_token.items()},
        "normalized_position_quartiles": position_quartiles,
    }


def exact_divergence_vocab_chunked(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor | None = None,
    *,
    divergence: str = "forward_kl",
    temperature: float = 1.0,
    reduction: str = "batchmean",
    chunk_size: int,
) -> torch.Tensor:
    """Compute a canonical full-vocabulary divergence in bounded memory."""
    if divergence not in {"forward_kl", "reverse_kl", "js"}:
        raise ValueError(f"unsupported divergence {divergence!r}")
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

    work_dtype = torch.float64 if student_logits.dtype == torch.float64 else torch.float32
    student_scaled = student_logits.to(work_dtype) / temperature
    teacher_scaled = teacher_logits.to(work_dtype) / temperature
    student_log_normalizer = torch.logsumexp(student_scaled, dim=-1, keepdim=True)
    teacher_log_normalizer = torch.logsumexp(teacher_scaled, dim=-1, keepdim=True)

    mask = None
    if labels is not None:
        if labels.shape != student_logits.shape[:2]:
            raise ValueError("labels must match the batch and sequence dimensions of logits")
        mask = labels != -100
        denominator = mask.sum()
        if int(denominator.item()) == 0:
            raise ValueError("cannot reduce an OPSD batch with no unmasked tokens")
    else:
        denominator = student_logits.new_tensor(student_logits.shape[0])

    total = student_scaled.new_zeros(())
    vocabulary_size = student_logits.shape[-1]
    for start in range(0, vocabulary_size, chunk_size):
        stop = min(start + chunk_size, vocabulary_size)
        student_log_probs = student_scaled[..., start:stop] - student_log_normalizer
        teacher_log_probs = teacher_scaled[..., start:stop] - teacher_log_normalizer
        if divergence == "forward_kl":
            probabilities = torch.exp(teacher_log_probs)
            log_ratio = teacher_log_probs - student_log_probs
            contributions = probabilities * log_ratio
        elif divergence == "reverse_kl":
            probabilities = torch.exp(student_log_probs)
            log_ratio = student_log_probs - teacher_log_probs
            contributions = probabilities * log_ratio
        else:
            log_two = student_log_probs.new_tensor(2.0).log()
            mixture_log_probs = torch.logaddexp(
                student_log_probs, teacher_log_probs
            ) - log_two
            contributions = 0.5 * (
                torch.exp(teacher_log_probs)
                * (teacher_log_probs - mixture_log_probs)
                + torch.exp(student_log_probs)
                * (student_log_probs - mixture_log_probs)
            )
        if mask is not None:
            contributions = contributions * mask.unsqueeze(-1)
        total = total + contributions.sum()
    return total / denominator


class _RecomputedDivergence(torch.autograd.Function):
    """Store only logits/mask; reconstruct vocabulary intermediates backward."""

    @staticmethod
    def forward(
        ctx,
        student_logits,
        teacher_logits,
        labels,
        has_labels,
        divergence,
        temperature,
        chunk_size,
    ):
        ctx.has_labels = bool(has_labels)
        ctx.divergence = str(divergence)
        ctx.temperature = float(temperature)
        ctx.chunk_size = int(chunk_size)
        mask = (
            labels != -100
            if ctx.has_labels
            else torch.ones(
                student_logits.shape[:2],
                dtype=torch.bool,
                device=student_logits.device,
            )
        )
        ctx.save_for_backward(student_logits, teacher_logits, mask)
        return exact_divergence_vocab_chunked(
            student_logits,
            teacher_logits,
            labels if ctx.has_labels else None,
            divergence=ctx.divergence,
            temperature=ctx.temperature,
            chunk_size=ctx.chunk_size,
        )

    @staticmethod
    def backward(ctx, output_gradient):
        student_logits, teacher_logits, mask = ctx.saved_tensors
        work_dtype = (
            torch.float64
            if student_logits.dtype == torch.float64
            else torch.float32
        )
        temperature = ctx.temperature
        student_scaled = student_logits.to(work_dtype) / temperature
        teacher_scaled = teacher_logits.to(work_dtype) / temperature
        student_log_normalizer = torch.logsumexp(
            student_scaled, dim=-1, keepdim=True
        )
        teacher_log_normalizer = torch.logsumexp(
            teacher_scaled, dim=-1, keepdim=True
        )
        denominator = (
            int(mask.sum().item())
            if ctx.has_labels
            else student_logits.shape[0]
        )
        vocabulary_size = student_logits.shape[-1]
        center = None
        if ctx.divergence in {"reverse_kl", "js"}:
            center = student_scaled.new_zeros(student_logits.shape[:2])
            for start in range(0, vocabulary_size, ctx.chunk_size):
                stop = min(start + ctx.chunk_size, vocabulary_size)
                student_log_probs = (
                    student_scaled[..., start:stop]
                    - student_log_normalizer
                )
                teacher_log_probs = (
                    teacher_scaled[..., start:stop]
                    - teacher_log_normalizer
                )
                if ctx.divergence == "reverse_kl":
                    log_ratio = student_log_probs - teacher_log_probs
                else:
                    mixture_log_probs = torch.logaddexp(
                        student_log_probs, teacher_log_probs
                    ) - student_log_probs.new_tensor(2.0).log()
                    log_ratio = student_log_probs - mixture_log_probs
                center += (
                    torch.exp(student_log_probs) * log_ratio
                ).sum(dim=-1)

        student_gradient = torch.empty_like(student_logits)
        mask_scale = mask.to(work_dtype).unsqueeze(-1)
        upstream = output_gradient.to(work_dtype)
        scale = upstream / (denominator * temperature)
        for start in range(0, vocabulary_size, ctx.chunk_size):
            stop = min(start + ctx.chunk_size, vocabulary_size)
            student_log_probs = (
                student_scaled[..., start:stop] - student_log_normalizer
            )
            teacher_log_probs = (
                teacher_scaled[..., start:stop] - teacher_log_normalizer
            )
            student_probs = torch.exp(student_log_probs)
            if ctx.divergence == "forward_kl":
                gradient = student_probs - torch.exp(teacher_log_probs)
            elif ctx.divergence == "reverse_kl":
                gradient = student_probs * (
                    student_log_probs
                    - teacher_log_probs
                    - center.unsqueeze(-1)
                )
            else:
                mixture_log_probs = torch.logaddexp(
                    student_log_probs, teacher_log_probs
                ) - student_log_probs.new_tensor(2.0).log()
                gradient = 0.5 * student_probs * (
                    student_log_probs
                    - mixture_log_probs
                    - center.unsqueeze(-1)
                )
            student_gradient[..., start:stop] = (
                gradient * mask_scale * scale
            ).to(student_logits.dtype)
        return student_gradient, None, None, None, None, None, None


def recomputed_divergence_vocab_chunked(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor | None = None,
    *,
    divergence: str = "forward_kl",
    temperature: float = 1.0,
    reduction: str = "batchmean",
    chunk_size: int,
) -> torch.Tensor:
    """Canonical loss with analytic, recomputed student-logit gradients.

    The forward value is identical to ``exact_divergence_vocab_chunked``.
    Backward reconstructs probabilities a vocabulary chunk at a time instead
    of retaining every FKL/RKL/JS intermediate from the forward graph.
    """
    if reduction != "batchmean":
        raise ValueError(
            "recomputed chunked OPSD loss currently requires batchmean reduction"
        )
    if not student_logits.requires_grad:
        return exact_divergence_vocab_chunked(
            student_logits,
            teacher_logits,
            labels,
            divergence=divergence,
            temperature=temperature,
            reduction=reduction,
            chunk_size=chunk_size,
        )
    labels_tensor = (
        labels
        if labels is not None
        else torch.empty(0, dtype=torch.long, device=student_logits.device)
    )
    return _RecomputedDivergence.apply(
        student_logits,
        teacher_logits,
        labels_tensor,
        labels is not None,
        divergence,
        temperature,
        chunk_size,
    )


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

    # A KL between the frozen base model and a freshly initialized LoRA model
    # is very small.  Performing ``p * (log p - log q)`` and its vocabulary
    # reduction in BF16 then suffers catastrophic cancellation: in practice
    # it can round a non-negative KL to zero (or even a small negative value).
    # Keep model activations in their requested dtype, but promote only this
    # numerically sensitive reduction to FP32.  Retain FP64 for reference
    # tests and callers that intentionally supplied it.
    work_dtype = torch.float64 if student_logits.dtype == torch.float64 else torch.float32
    student_scaled = student_logits.to(work_dtype) / temperature
    teacher_scaled = teacher_logits.to(work_dtype) / temperature
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

    total = student_scaled.new_zeros(())
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
