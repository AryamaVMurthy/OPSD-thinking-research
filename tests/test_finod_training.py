from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from opsd_research.finod_training import _aggregate_finod_metrics


class _TwoRankMirrorAccelerator:
    num_processes = 2

    def reduce(self, value, reduction):
        if reduction == "sum":
            return value * 2
        raise AssertionError(reduction)

    def gather(self, value):
        return torch.cat([value, value])


def test_finod_telemetry_is_token_weighted_and_reduced_across_ranks():
    trainer = SimpleNamespace(accelerator=_TwoRankMirrorAccelerator())
    mask = torch.tensor([[True, False], [True, True]])
    metrics = {
        "guide_energy": torch.tensor([[2.0, 99.0], [4.0, 6.0]]),
        "nuisance_energy": torch.tensor([[1.0, 99.0], [2.0, 3.0]]),
        "residual_energy": torch.tensor([[1.0, 99.0], [2.0, 3.0]]),
        "target_forward_kl": torch.tensor(
            [[0.0008, 99.0], [0.0025, 0.0015]]
        ),
        "target_reverse_kl": torch.tensor(
            [[0.001, 99.0], [0.003, 0.002]]
        ),
        "target_kl": torch.tensor([[0.001, 99.0], [0.003, 0.002]]),
        "effective_step_size": torch.tensor([[0.2, 99.0], [0.1, 0.2]]),
        "active_projection": torch.tensor([[True, False], [False, True]]),
        "guide_nuisance_alignment": torch.tensor([[1.0, 99.0], [-1.0, 2.0]]),
        "residual_nuisance_alignment": torch.tensor([[0.0, 99.0], [-1.0, 0.0]]),
    }

    result = _aggregate_finod_metrics(
        trainer,
        loss=torch.tensor(0.004),
        metrics=metrics,
        selected_mask=mask,
        step_size=0.25,
        residual_energy_threshold=1.5,
    )

    assert result["retained_tokens"] == 6
    assert result["loss"] == pytest.approx(0.004)
    assert result["guide_energy"] == pytest.approx(4.0)
    assert result["target_kl"] == pytest.approx(0.002)
    assert result["target_forward_kl"] == pytest.approx(0.0016)
    assert result["target_reverse_kl"] == pytest.approx(0.002)
    assert result["max_observed_target_kl"] == pytest.approx(0.003)
    assert result["active_projection_fraction"] == pytest.approx(2 / 3)
    assert result["collapsed_residual_fraction"] == pytest.approx(1 / 3)
    assert result["positive_alignment_after_fraction"] == 0.0
    assert result["clipped_target_fraction"] == 1.0
