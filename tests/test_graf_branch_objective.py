import pytest

torch = pytest.importorskip("torch", reason="routing objective requires PyTorch")

from opsd_research.graf_branch_objective import build_routed_action_batch
from opsd_research.graf_routing import ActionTarget, ForkTarget


class TinyTokenizer:
    """Deterministic tokenization sufficient to test routing tensor layout."""

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
    rows, target = batch.target_groups[0]
    assert rows.tolist() == [0, 1]
    torch.testing.assert_close(target, torch.tensor([0.75, 0.25]))
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
