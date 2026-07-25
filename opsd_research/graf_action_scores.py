"""Token-normalized action scores for the GRAF routed branch objective."""

from __future__ import annotations

import torch
import torch.nn.functional as functional


def action_scores_from_tail_logits(
    tail_logits: torch.Tensor,
    action_token_ids: torch.Tensor,
    action_lengths: torch.Tensor,
) -> torch.Tensor:
    """Return mean log-probability for each candidate action sequence.

    `tail_logits` contains `max_action_tokens + 1` final model positions. The
    final position predicts a token after the action and is intentionally
    discarded.  Left-aligned action ends let variable-length actions share one
    model forward without comparing raw sequence sums (which favor short text).
    """
    if tail_logits.ndim != 3:
        raise ValueError("tail_logits must have shape [actions, tokens, vocab]")
    action_count, tail_count, vocabulary_size = tail_logits.shape
    if action_token_ids.ndim != 2 or action_token_ids.shape[0] != action_count:
        raise ValueError("action_token_ids must have shape [actions, max_action_tokens]")
    if action_lengths.shape != (action_count,):
        raise ValueError("action_lengths must have shape [actions]")
    max_action_tokens = action_token_ids.shape[1]
    if tail_count != max_action_tokens + 1:
        raise ValueError("tail logits must include exactly one next-token position")
    if action_token_ids.dtype != torch.long:
        raise ValueError("action_token_ids must be torch.long")
    if torch.any(action_lengths < 1) or torch.any(action_lengths > max_action_tokens):
        raise ValueError("action lengths must be in [1, max_action_tokens]")
    if torch.any(action_token_ids < 0) or torch.any(action_token_ids >= vocabulary_size):
        raise ValueError("action token IDs are outside the vocabulary")

    log_probs = functional.log_softmax(tail_logits[:, :-1, :], dim=-1)
    positions = torch.arange(max_action_tokens, device=tail_logits.device).unsqueeze(0)
    starts = (max_action_tokens - action_lengths).unsqueeze(1)
    mask = positions >= starts
    token_log_probs = log_probs.gather(
        dim=-1, index=action_token_ids.unsqueeze(-1)
    ).squeeze(-1)
    return (token_log_probs * mask).sum(dim=-1) / action_lengths
