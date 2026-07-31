from __future__ import annotations

import torch

from opsd_research.finod import (
    fisher_projected_loss,
    fisher_projected_target,
    select_rollout_positions,
)


def test_projection_removes_answer_direction_but_retains_procedural_signal():
    student = torch.zeros(1, 3, dtype=torch.float64)
    base = torch.zeros_like(student)
    answer = torch.tensor([[1.0, -1.0, 0.0]], dtype=torch.float64)
    procedure = torch.tensor([[1.0, 1.0, -2.0]], dtype=torch.float64)
    guide = answer + procedure

    result = fisher_projected_target(
        student_logits=student,
        guide_logits=guide,
        base_logits=base,
        nuisance_logits=answer,
        step_size=0.25,
        nuisance_strength_threshold=1e-12,
    )

    expected = procedure
    torch.testing.assert_close(result.residual_direction, expected)
    assert result.metrics["active_projection"].item() == 1
    assert abs(result.metrics["residual_nuisance_alignment"].item()) < 1e-12
    assert result.metrics["guide_energy"].item() > result.metrics["residual_energy"].item()
    assert result.metrics["target_kl"].item() > 0.0
    assert not torch.allclose(result.target_probs, student.softmax(dim=-1))


def test_signed_projection_removes_anti_aligned_answer_direction():
    student = torch.zeros(1, 3, dtype=torch.float64)
    base = torch.zeros_like(student)
    answer = torch.tensor([[1.0, -1.0, 0.0]], dtype=torch.float64)
    procedure = torch.tensor([[1.0, 1.0, -2.0]], dtype=torch.float64)
    guide = procedure - answer

    result = fisher_projected_target(
        student_logits=student,
        guide_logits=guide,
        base_logits=base,
        nuisance_logits=answer,
        step_size=0.25,
        nuisance_strength_threshold=1e-12,
        projection_mode="signed-orthogonal-v1",
    )

    torch.testing.assert_close(result.residual_direction, procedure)
    assert result.metrics["projection_coefficient"].item() < 0.0
    assert abs(result.metrics["residual_nuisance_alignment"].item()) < 1e-12
    assert result.metrics["residual_energy"].item() > 0.0
    assert result.metrics["target_kl"].item() > 0.0


def test_target_kl_is_clipped_without_changing_residual_direction():
    student = torch.zeros(1, 4, dtype=torch.float64)
    base = torch.zeros_like(student)
    nuisance = torch.tensor([[1.0, -1.0, 0.0, 0.0]], dtype=torch.float64)
    guide = nuisance + torch.tensor(
        [[20.0, 20.0, -20.0, -20.0]], dtype=torch.float64
    )

    result = fisher_projected_target(
        student_logits=student,
        guide_logits=guide,
        base_logits=base,
        nuisance_logits=nuisance,
        step_size=1.0,
        max_target_kl=0.01,
        nuisance_strength_threshold=1e-12,
    )

    assert result.metrics["target_kl"].item() <= 0.0100001
    assert result.metrics["target_forward_kl"].item() <= 0.0100001
    assert result.metrics["target_reverse_kl"].item() <= 0.0100001
    assert abs(
        result.metrics["target_kl"].item()
        - max(
            result.metrics["target_forward_kl"].item(),
            result.metrics["target_reverse_kl"].item(),
        )
    ) < 1e-12
    assert result.metrics["effective_step_size"].item() < 1.0
    torch.testing.assert_close(
        result.residual_direction,
        torch.tensor([[20.0, 20.0, -20.0, -20.0]], dtype=torch.float64),
    )


def test_projected_distillation_has_nonzero_finite_student_gradient():
    student = torch.zeros(2, 3, dtype=torch.float64, requires_grad=True)
    base = torch.zeros_like(student)
    nuisance = torch.tensor(
        [[1.0, -1.0, 0.0], [1.0, -1.0, 0.0]], dtype=torch.float64
    )
    guide = nuisance + torch.tensor(
        [[1.0, 1.0, -2.0], [-1.0, -1.0, 2.0]], dtype=torch.float64
    )
    mask = torch.tensor([True, False])

    loss, result = fisher_projected_loss(
        student_logits=student,
        guide_logits=guide,
        base_logits=base,
        nuisance_logits=nuisance,
        token_mask=mask,
        step_size=0.25,
        max_target_kl=0.02,
        nuisance_strength_threshold=1e-12,
    )
    loss.backward()

    assert 0.0 < loss.item() <= 0.0200001
    assert result.metrics["retained_token_count"].item() == 1
    assert torch.isfinite(student.grad).all()
    assert student.grad[0].abs().sum().item() > 0.0
    assert student.grad[1].abs().sum().item() == 0.0


def test_sparse_positions_span_the_valid_rollout_and_ignore_padding():
    mask = torch.tensor(
        [
            [True, True, True, True, True, False, False],
            [True, True, True, False, False, False, False],
        ]
    )

    positions = select_rollout_positions(mask, max_positions=3)

    assert positions.tolist() == [0, 2, 4]
