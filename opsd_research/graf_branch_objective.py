"""Memory-bounded action routing loss for the GRAF-OPSD candidate.

The normal OPSD forward pass trains on an on-policy completion.  This module
adds a separate, short forward pass over *candidate next actions* and compares
their normalized likelihoods to frozen forced-continuation viability targets.
Neither targets nor graph text are added to the student prompt.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from .graf_actions import action_continuation
from .graf_action_scores import action_scores_from_tail_logits
from .graf_loss import branch_routed_loss
from .graf_routing import ForkTarget

@dataclass(frozen=True)
class RoutedActionBatch:
    """Left-padded sequences and per-fork row groups for one policy forward."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    action_token_ids: torch.Tensor
    action_lengths: torch.Tensor
    target_groups: tuple[tuple[torch.Tensor, torch.Tensor, float], ...]


def build_routed_action_batch(
    *,
    tokenizer,
    student_prompts: torch.Tensor,
    student_prompt_lengths: torch.Tensor,
    source_indices: torch.Tensor,
    routing_targets: Mapping[int, Sequence[ForkTarget]],
    pad_token_id: int,
    max_action_tokens: int = 128,
) -> RoutedActionBatch | None:
    """Materialize short candidate-action continuations without prompt padding.

    Candidate sequences are left-padded so the final ``max_action_tokens + 1``
    logits align across prompts of differing lengths.  A group corresponds to
    one graph fork and is averaged by the caller, preventing examples with two
    forks from receiving disproportionate weight.
    """
    if student_prompts.ndim != 2:
        raise ValueError("student_prompts must have shape [batch, tokens]")
    if student_prompt_lengths.shape != (student_prompts.shape[0],):
        raise ValueError("student_prompt_lengths must have one value per batch item")
    if source_indices.shape != (student_prompts.shape[0],):
        raise ValueError("source_indices must have one value per batch item")
    if max_action_tokens < 1:
        raise ValueError("max_action_tokens must be positive")

    sequences: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    targets: list[tuple[list[int], list[float], float]] = []
    for batch_index, raw_source in enumerate(source_indices.detach().cpu().tolist()):
        prompt_length = int(student_prompt_lengths[batch_index].item())
        prompt = student_prompts[batch_index, :prompt_length]
        for fork in routing_targets.get(int(raw_source), ()):
            encoded = [tokenizer(action_continuation(action.description), add_special_tokens=False)["input_ids"] for action in fork.actions]
            if any(not ids or len(ids) > max_action_tokens for ids in encoded):
                continue
            rows: list[int] = []
            probs: list[float] = []
            for ids, action in zip(encoded, fork.actions, strict=True):
                action_tokens = torch.tensor(ids, dtype=torch.long, device=student_prompts.device)
                sequences.append(torch.cat((prompt, action_tokens)))
                actions.append(action_tokens)
                rows.append(len(sequences) - 1)
                probs.append(float(action.target_probability))
            targets.append((rows, probs, float(fork.information_weight)))
    if not targets:
        return None

    max_sequence = max(sequence.numel() for sequence in sequences)
    max_action = max(action.numel() for action in actions)
    input_ids = torch.full(
        (len(sequences), max_sequence), pad_token_id, dtype=torch.long, device=student_prompts.device
    )
    attention_mask = torch.zeros_like(input_ids)
    action_token_ids = torch.full(
        (len(actions), max_action), pad_token_id, dtype=torch.long, device=student_prompts.device
    )
    action_lengths = torch.empty(len(actions), dtype=torch.long, device=student_prompts.device)
    for row, (sequence, action) in enumerate(zip(sequences, actions, strict=True)):
        input_ids[row, -sequence.numel() :] = sequence
        attention_mask[row, -sequence.numel() :] = 1
        action_token_ids[row, -action.numel() :] = action
        action_lengths[row] = action.numel()
    return RoutedActionBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        action_token_ids=action_token_ids,
        action_lengths=action_lengths,
        target_groups=tuple(
            (
                torch.tensor(rows, dtype=torch.long, device=student_prompts.device),
                torch.tensor(probabilities, dtype=torch.float32, device=student_prompts.device),
                information_weight,
            )
            for rows, probabilities, information_weight in targets
        ),
    )


def graf_branch_loss(
    *,
    model,
    inputs: Mapping[str, torch.Tensor],
    tokenizer,
    routing_targets: Mapping[int, Sequence[ForkTarget]],
    branch_loss_weight: float,
    entropy_floor_fraction: float,
    max_action_tokens: int = 128,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Score frozen graph actions and return a weighted, differentiable loss."""
    if branch_loss_weight <= 0:
        zero = inputs["student_input_ids"].sum() * 0.0
        return zero, {"active_forks": 0.0, "branch_kl": 0.0, "entropy_floor": 0.0}
    required = {"student_prompts", "student_prompt_lengths_per_example", "graf_source_index"}
    missing = required.difference(inputs)
    if missing:
        raise RuntimeError(f"GRAF routed loss missing batch fields: {sorted(missing)}")
    batch = build_routed_action_batch(
        tokenizer=tokenizer,
        student_prompts=inputs["student_prompts"],
        student_prompt_lengths=inputs["student_prompt_lengths_per_example"],
        source_indices=inputs["graf_source_index"],
        routing_targets=routing_targets,
        pad_token_id=int(tokenizer.pad_token_id),
        max_action_tokens=max_action_tokens,
    )
    if batch is None:
        zero = inputs["student_input_ids"].sum() * 0.0
        return zero, {"active_forks": 0.0, "branch_kl": 0.0, "entropy_floor": 0.0}
    outputs = model(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        logits_to_keep=batch.action_token_ids.shape[1] + 1,
    )
    scores = action_scores_from_tail_logits(
        outputs.logits, batch.action_token_ids, batch.action_lengths
    )
    losses: list[torch.Tensor] = []
    weights: list[float] = []
    branch_kl = 0.0
    entropy_floor = 0.0
    for rows, target, information_weight in batch.target_groups:
        fork_loss, metrics = branch_routed_loss(
            scores.index_select(0, rows).unsqueeze(0),
            target.unsqueeze(0),
            torch.tensor([True], dtype=torch.bool, device=scores.device),
            entropy_floor_fraction=entropy_floor_fraction,
        )
        losses.append(fork_loss)
        weights.append(information_weight)
        branch_kl += float(metrics["branch_kl"]) * information_weight
        entropy_floor += float(metrics["entropy_floor"]) * information_weight
    weight_sum = sum(weights)
    # Information weights represent absolute evidence strength. Dividing by
    # ``weight_sum`` made the weight cancel whenever a microbatch had one fork,
    # so a barely informative target received the same gradient as a certain
    # one. Average over the number of opportunities instead, preserving the
    # configured confidence attenuation.
    mean_loss = sum(
        loss * weight for loss, weight in zip(losses, weights, strict=True)
    ) / len(losses)
    active = float(len(losses))
    return branch_loss_weight * mean_loss, {
        "active_forks": active,
        "effective_fork_weight": weight_sum,
        "branch_kl": branch_kl / weight_sum,
        "entropy_floor": entropy_floor / weight_sum,
    }
