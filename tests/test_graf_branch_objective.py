import unittest

try:
    import torch
except ImportError as error:
    raise unittest.SkipTest("routing objective requires PyTorch") from error

from opsd_research.graf_branch_objective import (
    build_routed_action_batch,
    graf_branch_loss,
)
from opsd_research.graf_routing import ActionTarget, ForkTarget


class TinyTokenizer:
    """Deterministic tokenization sufficient to test routing tensor layout."""

    pad_token_id = 0

    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": [2 + (ord(char) % 23) for char in text[-3:]]}


def test_builds_left_padded_action_continuations_and_groups() -> None:
    targets = {
        7: (
            ForkTarget(
                fork_id="f0",
                actions=(
                    ActionTarget("a0", "factor", 0.75),
                    ActionTarget("a1", "substitute", 0.25),
                ),
                information_weight=0.4,
            ),
        )
    }
    batch = build_routed_action_batch(
        tokenizer=TinyTokenizer(),
        student_prompts=torch.tensor([[4, 5, 0], [6, 7, 8]]),
        student_prompt_lengths=torch.tensor([2, 3]),
        source_indices=torch.tensor([7, 99]),
        routing_targets=targets,
        pad_token_id=0,
        max_action_tokens=8,
    )
    assert batch is not None
    assert batch.input_ids.shape[0] == 2
    assert batch.action_token_ids.shape[0] == 2
    assert len(batch.target_groups) == 1
    rows, target, information_weight = batch.target_groups[0]
    assert rows.tolist() == [0, 1]
    torch.testing.assert_close(target, torch.tensor([0.75, 0.25]))
    assert information_weight == 0.4
    # Both full sequences are left-padded to a common terminal action boundary.
    assert torch.all(batch.attention_mask[:, -batch.action_lengths.max() :] == 1)


def test_skips_an_entire_fork_when_an_action_is_too_long() -> None:
    targets = {
        7: (
            ForkTarget(
                fork_id="f0",
                actions=(ActionTarget("a0", "factor", 0.5), ActionTarget("a1", "substitute", 0.5)),
            ),
        )
    }
    batch = build_routed_action_batch(
        tokenizer=TinyTokenizer(),
        student_prompts=torch.tensor([[4, 5]]),
        student_prompt_lengths=torch.tensor([2]),
        source_indices=torch.tensor([7]),
        routing_targets=targets,
        pad_token_id=0,
        max_action_tokens=1,
    )
    assert batch is None


def test_single_fork_information_weight_scales_its_absolute_gradient() -> None:
    class UniformModel:
        def __call__(self, *, input_ids, attention_mask, logits_to_keep):
            del attention_mask
            logits = torch.zeros(
                input_ids.shape[0],
                logits_to_keep,
                32,
                dtype=torch.float32,
                requires_grad=True,
            )
            return type("Output", (), {"logits": logits})()

    inputs = {
        "student_input_ids": torch.tensor([[4, 5]]),
        "student_prompts": torch.tensor([[4, 5]]),
        "student_prompt_lengths_per_example": torch.tensor([2]),
        "graf_source_index": torch.tensor([7]),
    }

    def targets(weight: float):
        return {
            7: (
                ForkTarget(
                    fork_id="f0",
                    actions=(
                        ActionTarget("a0", "factor", 0.75),
                        ActionTarget("a1", "substitute", 0.25),
                    ),
                    information_weight=weight,
                ),
            )
        }

    full, _ = graf_branch_loss(
        model=UniformModel(),
        inputs=inputs,
        tokenizer=TinyTokenizer(),
        routing_targets=targets(1.0),
        branch_loss_weight=1.0,
        entropy_floor_fraction=0.0,
    )
    uncertain, metrics = graf_branch_loss(
        model=UniformModel(),
        inputs=inputs,
        tokenizer=TinyTokenizer(),
        routing_targets=targets(0.25),
        branch_loss_weight=1.0,
        entropy_floor_fraction=0.0,
    )

    torch.testing.assert_close(uncertain, 0.25 * full)
    assert metrics["effective_fork_weight"] == 0.25
