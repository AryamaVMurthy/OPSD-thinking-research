import unittest

import pytest

torch = pytest.importorskip("torch", reason="loss unit test requires PyTorch")

from opsd_research.graf_loss import branch_routed_loss


class GrafLossTests(unittest.TestCase):
    def test_branch_loss_has_gradient_only_for_aligned_forks(self):
        scores = torch.tensor([[2.0, 0.0], [0.0, 0.0]], requires_grad=True)
        target = torch.tensor([[0.75, 0.25], [0.5, 0.5]])
        loss, metrics = branch_routed_loss(scores, target, torch.tensor([True, False]), entropy_floor_fraction=0.5)
        loss.backward()
        self.assertGreater(float(scores.grad[0].abs().sum()), 0.0)
        self.assertEqual(float(scores.grad[1].abs().sum()), 0.0)
        self.assertEqual(float(metrics["active_forks"]), 1.0)

    def test_empty_fork_mask_has_zero_loss(self):
        scores = torch.zeros((2, 2), requires_grad=True)
        target = torch.full((2, 2), 0.5)
        loss, _ = branch_routed_loss(scores, target, torch.tensor([False, False]))
        loss.backward()
        self.assertEqual(float(loss), 0.0)
