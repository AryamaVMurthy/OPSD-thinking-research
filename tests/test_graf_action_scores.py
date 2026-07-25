import pytest

torch = pytest.importorskip("torch", reason="action score unit test requires PyTorch")

from opsd_research.graf_action_scores import action_scores_from_tail_logits


def test_scores_variable_length_actions_without_short_sequence_bias() -> None:
    # Two actions share the same per-token likelihood; their normalized scores
    # must match even though the second has two tokens.
    logits = torch.tensor([
        [[0.0, 2.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.0, 2.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]],
    ])
    tokens = torch.tensor([[0, 1], [1, 1]])
    lengths = torch.tensor([1, 2])

    scores = action_scores_from_tail_logits(logits, tokens, lengths)

    assert torch.allclose(scores[0], scores[1])


def test_rejects_tail_without_next_token_position() -> None:
    with pytest.raises(ValueError, match="next-token"):
        action_scores_from_tail_logits(
            torch.zeros((1, 2, 3)), torch.zeros((1, 2), dtype=torch.long), torch.ones(1, dtype=torch.long)
        )
