from __future__ import annotations

import pytest
import torch

from opsd_research.fisher_consensus import (
    fisher_edge_boost_weights,
    fisher_score_signature,
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


def test_fisher_score_signature_uses_whitened_vocabulary_coordinates():
    base = torch.zeros((1, 2, 3), dtype=torch.float64)
    direction = torch.tensor(
        [[[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]],
        dtype=torch.float64,
    )
    mask = torch.tensor([[True, False]])

    signature, norms = fisher_score_signature(
        base_logits=base,
        consensus_direction=direction,
        token_mask=mask,
    )

    expected = direction[:, 0] / (3.0**0.5)
    assert torch.allclose(signature, expected)
    assert norms.item() == pytest.approx(expected.norm().item())


def test_identical_cross_problem_signatures_have_unit_edge_and_weight():
    signatures = torch.tensor(
        [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]],
        dtype=torch.float64,
    )

    result = fisher_edge_boost_weights(signatures, threshold=0.0)

    assert torch.allclose(
        result.signed_edges, torch.ones_like(result.signed_edges)
    )
    assert torch.allclose(result.weights, torch.ones_like(result.weights))
    assert result.active_fraction.item() == pytest.approx(1.0)


def test_leave_one_out_edge_removes_positive_self_similarity():
    signatures = torch.eye(3, dtype=torch.float64)

    result = fisher_edge_boost_weights(signatures, threshold=0.0)

    assert torch.allclose(
        result.signed_edges, torch.zeros_like(result.signed_edges)
    )
    assert torch.equal(result.weights, torch.zeros_like(result.weights))
    assert result.active_fraction.item() == pytest.approx(0.0)


def test_opposing_cross_problem_signatures_have_no_positive_edge():
    signatures = torch.tensor(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]],
        dtype=torch.float64,
    )

    result = fisher_edge_boost_weights(signatures, threshold=0.0)

    assert bool((result.signed_edges <= 0.0).all())
    assert torch.equal(result.weights, torch.zeros_like(result.weights))


def test_positive_fisher_edges_are_normalized_to_global_mean_one():
    signatures = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        dtype=torch.float64,
    )

    result = fisher_edge_boost_weights(signatures, threshold=0.0)

    assert result.active_fraction.item() == pytest.approx(2 / 3)
    assert result.weights.mean().item() == pytest.approx(1.0)
    assert result.weights[-1].item() == pytest.approx(0.0)


def test_positive_plan_barycenter_is_invariant_to_control_logits():
    base = _logits([0.2, -0.1, 0.0])
    guides = torch.tensor(
        [
            [[1.0, -1.0, 0.0]],
            [[0.8, -0.7, -0.1]],
            [[1.2, -0.9, -0.3]],
        ],
        dtype=torch.float64,
    )
    controls_a = torch.zeros_like(guides)
    controls_b = torch.randn_like(guides) * 100.0

    first = fisher_consensus_target(
        base_logits=base,
        guide_logits=guides,
        control_logits=controls_a,
        step_size=1.0,
        max_target_kl=0.01,
        direction_mode="positive_plan_barycenter",
    )
    second = fisher_consensus_target(
        base_logits=base,
        guide_logits=guides,
        control_logits=controls_b,
        step_size=1.0,
        max_target_kl=0.01,
        direction_mode="positive_plan_barycenter",
    )

    assert torch.equal(first.target_probs, second.target_probs)
    assert torch.equal(
        first.consensus_direction,
        second.consensus_direction,
    )


def test_positive_plan_barycenter_centers_guide_minus_base_scores():
    base = _logits([0.2, -0.1, 0.0])
    positive = _logits([0.8, -0.4, 0.1])
    guides = torch.stack([positive, positive, positive])

    result = fisher_consensus_target(
        base_logits=base,
        guide_logits=guides,
        control_logits=torch.zeros_like(guides),
        step_size=0.25,
        max_target_kl=0.01,
        direction_mode="positive_plan_barycenter",
    )

    expected = positive - base
    probability = base.softmax(dim=-1)
    expected -= (probability * expected).sum(dim=-1, keepdim=True)
    assert result.metrics["agreement"].item() == pytest.approx(1.0)
    assert torch.allclose(result.consensus_direction, expected)


def test_entropy_neutral_plan_barycenter_removes_entropy_tangent_only():
    base = _logits([2.0, 0.5, -0.5, -1.0])
    probability = base.softmax(dim=-1)
    log_probability = base.log_softmax(dim=-1)
    entropy_direction = log_probability - (
        probability * log_probability
    ).sum(dim=-1, keepdim=True)
    procedure = _logits([0.0, 1.0, -2.0, 1.0])
    procedure -= (probability * procedure).sum(dim=-1, keepdim=True)
    procedure -= (
        (probability * procedure * entropy_direction).sum(
            dim=-1, keepdim=True
        )
        / (probability * entropy_direction.square()).sum(
            dim=-1, keepdim=True
        )
    ) * entropy_direction
    positive = base + procedure + 3.0 * entropy_direction
    guides = torch.stack([positive, positive, positive])

    result = fisher_consensus_target(
        base_logits=base,
        guide_logits=guides,
        control_logits=torch.randn_like(guides) * 100.0,
        step_size=1.0,
        max_target_kl=0.01,
        direction_mode="entropy_neutral_plan_barycenter",
    )

    alignment = (
        probability * result.consensus_direction * entropy_direction
    ).sum(dim=-1)
    centered = (probability * result.consensus_direction).sum(dim=-1)
    assert alignment.item() == pytest.approx(0.0, abs=1e-12)
    assert centered.item() == pytest.approx(0.0, abs=1e-12)
    assert torch.allclose(result.consensus_direction, procedure)
    assert result.metrics["entropy_alignment_before"].abs().item() > 0.0
    assert result.metrics["entropy_alignment_after"].item() == pytest.approx(
        0.0, abs=1e-12
    )
    assert 0.0 < result.metrics["retained_direction_energy_fraction"].item() < 1.0
    assert result.metrics["target_kl"].item() <= 0.01000001


def test_entropy_neutral_barycenter_is_identity_at_uniform_anchor():
    base = _logits([0.0, 0.0, 0.0])
    guides = torch.tensor(
        [
            [[1.0, -1.0, 0.0]],
            [[0.8, -0.7, -0.1]],
            [[1.2, -0.9, -0.3]],
        ],
        dtype=torch.float64,
    )
    controls = torch.randn_like(guides)

    ordinary = fisher_consensus_target(
        base_logits=base,
        guide_logits=guides,
        control_logits=controls,
        step_size=1.0,
        max_target_kl=0.01,
        direction_mode="positive_plan_barycenter",
    )
    neutral = fisher_consensus_target(
        base_logits=base,
        guide_logits=guides,
        control_logits=controls,
        step_size=1.0,
        max_target_kl=0.01,
        direction_mode="entropy_neutral_plan_barycenter",
    )

    assert neutral.metrics["entropy_gradient_energy"].item() == pytest.approx(
        0.0
    )
    assert torch.equal(neutral.consensus_direction, ordinary.consensus_direction)
    assert torch.equal(neutral.target_probs, ordinary.target_probs)


def test_entropy_neutral_barycenter_is_invariant_to_control_logits():
    base = _logits([1.0, 0.1, -0.4])
    guides = torch.tensor(
        [
            [[1.4, -0.2, 0.1]],
            [[1.1, 0.0, -0.3]],
            [[1.3, -0.1, -0.2]],
        ],
        dtype=torch.float64,
    )

    first = fisher_consensus_target(
        base_logits=base,
        guide_logits=guides,
        control_logits=torch.zeros_like(guides),
        step_size=1.0,
        max_target_kl=0.01,
        direction_mode="entropy_neutral_plan_barycenter",
    )
    second = fisher_consensus_target(
        base_logits=base,
        guide_logits=guides,
        control_logits=torch.randn_like(guides) * 1000.0,
        step_size=1.0,
        max_target_kl=0.01,
        direction_mode="entropy_neutral_plan_barycenter",
    )

    assert torch.equal(first.consensus_direction, second.consensus_direction)
    assert torch.equal(first.target_probs, second.target_probs)


def test_entropy_neutral_barycenter_preserves_an_already_neutral_direction():
    base = _logits([2.0, 0.5, -0.5, -1.0])
    probability = base.softmax(dim=-1)
    log_probability = base.log_softmax(dim=-1)
    entropy_direction = log_probability - (
        probability * log_probability
    ).sum(dim=-1, keepdim=True)
    procedure = _logits([0.0, 1.0, -2.0, 1.0])
    procedure -= (probability * procedure).sum(dim=-1, keepdim=True)
    procedure -= (
        (probability * procedure * entropy_direction).sum(
            dim=-1, keepdim=True
        )
        / (probability * entropy_direction.square()).sum(
            dim=-1, keepdim=True
        )
    ) * entropy_direction
    guides = torch.stack([base + procedure] * 3)
    controls = torch.randn_like(guides)

    ordinary = fisher_consensus_target(
        base_logits=base,
        guide_logits=guides,
        control_logits=controls,
        step_size=1.0,
        max_target_kl=0.01,
        direction_mode="positive_plan_barycenter",
    )
    neutral = fisher_consensus_target(
        base_logits=base,
        guide_logits=guides,
        control_logits=controls,
        step_size=1.0,
        max_target_kl=0.01,
        direction_mode="entropy_neutral_plan_barycenter",
    )

    assert torch.allclose(
        neutral.consensus_direction,
        ordinary.consensus_direction,
        atol=1e-12,
        rtol=0.0,
    )
    assert torch.allclose(
        neutral.target_probs,
        ordinary.target_probs,
        atol=1e-12,
        rtol=0.0,
    )


def test_entropy_neutral_barycenter_rejects_nonfinite_active_logits():
    base = _logits([1.0, 0.0, -1.0])
    guides = torch.stack([base.clone(), base.clone(), base.clone()])
    guides[1, 0, 1] = torch.nan

    with pytest.raises(ValueError, match="nonfinite"):
        fisher_consensus_target(
            base_logits=base,
            guide_logits=guides,
            control_logits=torch.zeros_like(guides),
            step_size=1.0,
            max_target_kl=0.01,
            direction_mode="entropy_neutral_plan_barycenter",
        )


def test_entropy_neutral_barycenter_rejects_nonfinite_trust_region():
    base = _logits([1.0, 0.0, -1.0])
    guides = torch.stack([base.clone(), base.clone(), base.clone()])

    with pytest.raises(ValueError, match="finite"):
        fisher_consensus_target(
            base_logits=base,
            guide_logits=guides,
            control_logits=torch.zeros_like(guides),
            step_size=float("nan"),
            max_target_kl=0.01,
            direction_mode="entropy_neutral_plan_barycenter",
        )
