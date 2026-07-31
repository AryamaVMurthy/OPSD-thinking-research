from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from opsd_research.fisher_consensus_training import (
    _aggregate_consensus_metrics,
    _distributed_fisher_edge,
)


def test_consensus_metrics_are_token_weighted_and_report_collapse():
    metrics = {
        "agreement": torch.tensor([[1.0, 99.0], [0.5, 0.0]]),
        "mean_direction_energy": torch.tensor(
            [[2.0, 99.0], [4.0, 8.0]]
        ),
        "mean_pair_energy": torch.tensor(
            [[3.0, 99.0], [6.0, 9.0]]
        ),
        "consensus_energy": torch.tensor(
            [[2.0, 99.0], [1.0, 0.0]]
        ),
        "mean_pair_cosine": torch.tensor(
            [[1.0, 99.0], [0.7, 0.0]]
        ),
        "min_pair_cosine": torch.tensor(
            [[1.0, 99.0], [0.2, -0.1]]
        ),
        "target_forward_kl": torch.tensor(
            [[0.01, 99.0], [0.005, 0.0]]
        ),
        "target_reverse_kl": torch.tensor(
            [[0.009, 99.0], [0.004, 0.0]]
        ),
        "target_kl": torch.tensor(
            [[0.01, 99.0], [0.005, 0.0]]
        ),
        "effective_step_size": torch.tensor(
            [[0.2, 99.0], [0.25, 0.25]]
        ),
        "student_anchor_forward_kl": torch.tensor(
            [[0.02, 99.0], [0.01, 0.0]]
        ),
        "student_anchor_reverse_kl": torch.tensor(
            [[0.018, 99.0], [0.009, 0.0]]
        ),
        "target_student_kl": torch.tensor(
            [[0.03, 99.0], [0.02, 0.0]]
        ),
        "fisher_alignment_gain": torch.tensor(
            [[0.005, 99.0], [-0.002, 0.0]]
        ),
        "fisher_alignment_cosine_proxy": torch.tensor(
            [[0.4, 99.0], [-0.2, 0.0]]
        ),
        "optimization_per_token": torch.tensor(
            [[0.07, 99.0], [0.04, 0.0]]
        ),
    }
    selected = torch.tensor([[True, False], [True, True]])
    trainer = SimpleNamespace(
        accelerator=SimpleNamespace(),
    )

    result = _aggregate_consensus_metrics(
        trainer,
        loss=torch.tensor(0.3),
        metrics=metrics,
        selected_mask=selected,
        step_size=0.25,
        consensus_energy_threshold=1e-8,
    )

    assert result["retained_tokens"] == 3
    expected_target_loss = (0.03 + 0.02 + 0.0) / 3
    assert result["loss"] == pytest.approx(expected_target_loss)
    assert result["agreement"] == pytest.approx(0.5)
    assert result["collapsed_consensus_fraction"] == pytest.approx(1 / 3)
    assert result["clipped_target_fraction"] == pytest.approx(1 / 3)
    assert result["max_observed_target_kl"] == pytest.approx(0.01)
    expected_anchor_loss = (0.009 + 0.004 + 0.0) / 3
    assert result["anchor_target_loss"] == pytest.approx(
        expected_anchor_loss
    )
    assert result["relative_loss_to_anchor"] == pytest.approx(
        expected_target_loss / expected_anchor_loss
    )
    assert result["improvement_over_anchor"] == pytest.approx(
        expected_anchor_loss - expected_target_loss
    )
    assert result["student_anchor_forward_kl"] == pytest.approx(0.01)
    assert result["student_anchor_reverse_kl"] == pytest.approx(0.009)
    assert result["fisher_alignment_gain"] == pytest.approx(0.001)
    assert result["fisher_alignment_cosine_proxy"] == pytest.approx(
        0.2 / 3
    )
    assert result["optimization_loss"] == pytest.approx(0.11 / 3)


def test_distributed_fisher_edge_returns_the_local_rank_weight():
    gathered = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        dtype=torch.float32,
    )

    class Accelerator:
        process_index = 1
        num_processes = 3

        @staticmethod
        def gather(_signature):
            return gathered

    local_weight, result = _distributed_fisher_edge(
        SimpleNamespace(accelerator=Accelerator()),
        torch.tensor([[1.0, 0.0]]),
        threshold=0.0,
    )

    assert local_weight.shape == (1,)
    assert local_weight.item() == pytest.approx(1.5)
    assert result.active_fraction.item() == pytest.approx(2 / 3)
