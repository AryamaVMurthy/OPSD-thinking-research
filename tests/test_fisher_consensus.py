from __future__ import annotations

import pytest
import torch

from opsd_research.fisher_consensus import (
    fisher_consensus_loss,
    fisher_consensus_target,
)


def _logits(*rows: list[float]) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.float64)


def test_identical_contrastive_guides_have_unit_agreement():
    base = _logits([0.2, -0.1, 0.0])
    positive = _logits([0.8, -0.4, 0.1])
    negative = _logits([0.1, 0.0, -0.2])
    guides = torch.stack([positive, positive, positive])
    controls = torch.stack([negative, negative, negative])

    result = fisher_consensus_target(
        base_logits=base,
        guide_logits=guides,
        control_logits=controls,
        step_size=0.25,
        max_target_kl=0.01,
    )

    assert result.metrics["agreement"].item() == pytest.approx(1.0)
    expected = guides[0] - controls[0]
    probability = base.softmax(dim=-1)
    expected -= (probability * expected).sum(dim=-1, keepdim=True)
    assert torch.allclose(result.consensus_direction, expected)


def test_opposing_guides_cancel_in_fisher_space():
    base = _logits([0.0, 0.0, 0.0])
    positive = torch.tensor(
        [
            [[1.0, -1.0, 0.0]],
            [[-1.0, 1.0, 0.0]],
        ],
        dtype=torch.float64,
    )
    controls = torch.zeros_like(positive)

    result = fisher_consensus_target(
        base_logits=base,
        guide_logits=positive,
        control_logits=controls,
        step_size=0.25,
        max_target_kl=0.01,
    )

    assert result.metrics["agreement"].item() == pytest.approx(0.0)
    assert torch.equal(result.target_probs, base.softmax(dim=-1))


def test_agreement_is_bounded_and_scales_consensus_not_anchor():
    base = _logits([0.4, -0.2, 0.1, 0.0])
    guides = torch.tensor(
        [
            [[0.9, -0.3, 0.2, -0.1]],
            [[0.7, -0.2, 0.0, 0.1]],
            [[-0.1, 0.3, 0.2, 0.0]],
        ],
        dtype=torch.float64,
    )
    controls = torch.zeros_like(guides)

    result = fisher_consensus_target(
        base_logits=base,
        guide_logits=guides,
        control_logits=controls,
        step_size=0.25,
        max_target_kl=0.01,
    )

    agreement = result.metrics["agreement"].item()
    assert 0.0 <= agreement <= 1.0
    raw_mean = result.metrics["raw_mean_direction"]
    assert torch.allclose(result.consensus_direction, agreement * raw_mean)
    assert not torch.allclose(result.target_probs, base.softmax(dim=-1))


def test_two_sided_target_kl_is_clipped():
    base = _logits([0.0, 0.0, 0.0])
    guides = torch.tensor(
        [
            [[100.0, -100.0, 0.0]],
            [[100.0, -100.0, 0.0]],
            [[100.0, -100.0, 0.0]],
        ],
        dtype=torch.float64,
    )
    controls = torch.zeros_like(guides)

    result = fisher_consensus_target(
        base_logits=base,
        guide_logits=guides,
        control_logits=controls,
        step_size=1.0,
        max_target_kl=0.01,
    )

    assert result.metrics["target_forward_kl"].item() <= 0.01000001
    assert result.metrics["target_reverse_kl"].item() <= 0.01000001
    assert result.metrics["effective_step_size"].item() < 1.0


def test_loss_is_positive_and_backpropagates_only_to_student():
    student = _logits([0.2, -0.1, 0.0]).requires_grad_()
    base = _logits([0.0, 0.0, 0.0])
    guides = torch.tensor(
        [
            [[1.0, -1.0, 0.0]],
            [[0.8, -0.7, -0.1]],
            [[1.2, -0.9, -0.3]],
        ],
        dtype=torch.float64,
    )
    controls = torch.zeros_like(guides)

    loss, result = fisher_consensus_loss(
        student_logits=student,
        base_logits=base,
        guide_logits=guides,
        control_logits=controls,
        token_mask=torch.tensor([True]),
        step_size=0.25,
        max_target_kl=0.01,
    )
    loss.backward()

    assert loss.item() > 0.0
    assert student.grad is not None
    assert student.grad.abs().sum().item() > 0.0
    assert result.target_probs.requires_grad is False


def test_student_anchor_and_alignment_diagnostics_have_expected_endpoints():
    base = _logits([0.0, 0.0, 0.0])
    guides = torch.tensor(
        [
            [[1.0, -1.0, 0.0]],
            [[0.8, -0.7, -0.1]],
            [[1.2, -0.9, -0.3]],
        ],
        dtype=torch.float64,
    )
    controls = torch.zeros_like(guides)
    mask = torch.tensor([True])

    _, initial = fisher_consensus_loss(
        student_logits=base,
        base_logits=base,
        guide_logits=guides,
        control_logits=controls,
        token_mask=mask,
        step_size=0.25,
        max_target_kl=0.01,
    )
    assert initial.metrics["student_anchor_forward_kl"].item() == pytest.approx(
        0.0
    )
    assert initial.metrics["fisher_alignment_gain"].item() == pytest.approx(
        0.0
    )

    _, fitted = fisher_consensus_loss(
        student_logits=initial.target_probs.log(),
        base_logits=base,
        guide_logits=guides,
        control_logits=controls,
        token_mask=mask,
        step_size=0.25,
        max_target_kl=0.01,
    )
    assert fitted.metrics["student_anchor_forward_kl"].item() > 0.0
    assert fitted.metrics["fisher_alignment_gain"].item() > 0.0
    assert fitted.metrics["fisher_alignment_cosine_proxy"].item() == (
        pytest.approx(1.0)
    )


def test_anchor_regularizer_adds_frozen_policy_kl_to_optimization_loss():
    base = _logits([0.0, 0.0, 0.0])
    student = _logits([0.6, -0.2, -0.4])
    guides = torch.tensor(
        [
            [[1.0, -1.0, 0.0]],
            [[0.8, -0.7, -0.1]],
            [[1.2, -0.9, -0.3]],
        ],
        dtype=torch.float64,
    )
    controls = torch.zeros_like(guides)
    mask = torch.tensor([True])

    target_only, _ = fisher_consensus_loss(
        student_logits=student,
        base_logits=base,
        guide_logits=guides,
        control_logits=controls,
        token_mask=mask,
        step_size=0.25,
        max_target_kl=0.01,
        anchor_kl_weight=0.0,
    )
    proximal, result = fisher_consensus_loss(
        student_logits=student,
        base_logits=base,
        guide_logits=guides,
        control_logits=controls,
        token_mask=mask,
        step_size=0.25,
        max_target_kl=0.01,
        anchor_kl_weight=2.0,
    )

    anchor_kl = result.metrics["student_anchor_forward_kl"][mask].mean()
    assert proximal.item() == pytest.approx(
        target_only.item() + 2.0 * anchor_kl.item()
    )
    assert result.metrics["optimization_per_token"].item() == pytest.approx(
        proximal.item()
    )


def test_requires_at_least_two_matched_guide_control_pairs():
    base = _logits([0.0, 0.0])
    guides = torch.zeros((1, 1, 2), dtype=torch.float64)

    with pytest.raises(ValueError, match="at least two"):
        fisher_consensus_target(
            base_logits=base,
            guide_logits=guides,
            control_logits=guides,
            step_size=0.25,
            max_target_kl=0.01,
        )
