from __future__ import annotations

import os
import sys
import types
import unittest
from unittest import mock

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None

if torch is not None:
    from opsd_research import launch_official_opsd
    from opsd_research.jsd import (
        canonical_negative_tolerance,
        divergence_statistics_vocab_chunked,
        exact_divergence_vocab_chunked,
        exact_forward_kl_vocab_chunked,
        recomputed_divergence_vocab_chunked,
    )


def official_beta_zero_loss(
    student_logits,
    teacher_logits,
    labels,
    *,
    temperature: float,
    token_clip: float | None,
):
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)
    divergence = F.kl_div(
        student_log_probs,
        teacher_log_probs,
        reduction="none",
        log_target=True,
    )
    if token_clip is not None:
        divergence = divergence.clamp(max=token_clip)
    mask = labels != -100
    return divergence[mask].sum() / mask.sum()


@unittest.skipIf(torch is None, "torch is available in the pinned Turing environment")
class ChunkedJSDTests(unittest.TestCase):
    def test_public_forward_kl_matches_reference_without_clipping(self):
        generator = torch.Generator().manual_seed(3)
        student = torch.randn(
            2, 3, 11, generator=generator, dtype=torch.float64, requires_grad=True
        )
        teacher = torch.randn(2, 3, 11, generator=generator, dtype=torch.float64)
        labels = torch.tensor([[1, -100, 2], [3, 4, -100]], dtype=torch.long)

        student_log_probs = F.log_softmax(student / 0.8, dim=-1)
        teacher_log_probs = F.log_softmax(teacher / 0.8, dim=-1)
        per_token = F.kl_div(
            student_log_probs, teacher_log_probs, reduction="none", log_target=True
        ).sum(dim=-1)
        reference = per_token[labels != -100].mean()

        observed = exact_divergence_vocab_chunked(
            student,
            teacher,
            labels,
            divergence="forward_kl",
            temperature=0.8,
            chunk_size=4,
        )

        torch.testing.assert_close(observed, reference, rtol=1e-12, atol=1e-12)

    def test_public_reverse_kl_value_and_student_gradient_match_reference(self):
        generator = torch.Generator().manual_seed(5)
        student_reference = torch.randn(
            2, 3, 13, generator=generator, dtype=torch.float64, requires_grad=True
        )
        student_chunked = student_reference.detach().clone().requires_grad_(True)
        teacher = torch.randn(2, 3, 13, generator=generator, dtype=torch.float64)
        labels = torch.tensor([[1, 2, -100], [-100, 3, 4]], dtype=torch.long)

        student_log_probs = F.log_softmax(student_reference / 1.2, dim=-1)
        teacher_log_probs = F.log_softmax(teacher / 1.2, dim=-1)
        per_token = (
            student_log_probs.exp() * (student_log_probs - teacher_log_probs)
        ).sum(dim=-1)
        reference = per_token[labels != -100].mean()
        observed = exact_divergence_vocab_chunked(
            student_chunked,
            teacher,
            labels,
            divergence="reverse_kl",
            temperature=1.2,
            chunk_size=5,
        )
        reference.backward()
        observed.backward()

        torch.testing.assert_close(observed, reference, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(
            student_chunked.grad, student_reference.grad, rtol=1e-12, atol=1e-12
        )

    def test_public_js_value_and_student_gradient_match_reference(self):
        generator = torch.Generator().manual_seed(9)
        student_reference = torch.randn(
            2, 2, 19, generator=generator, dtype=torch.float64, requires_grad=True
        )
        student_chunked = student_reference.detach().clone().requires_grad_(True)
        teacher = torch.randn(2, 2, 19, generator=generator, dtype=torch.float64)
        labels = torch.tensor([[1, -100], [2, 3]], dtype=torch.long)

        student_log_probs = F.log_softmax(student_reference, dim=-1)
        teacher_log_probs = F.log_softmax(teacher, dim=-1)
        mixture_log_probs = torch.logaddexp(
            student_log_probs, teacher_log_probs
        ) - torch.log(torch.tensor(2.0, dtype=torch.float64))
        per_token = 0.5 * (
            teacher_log_probs.exp() * (teacher_log_probs - mixture_log_probs)
        ).sum(dim=-1) + 0.5 * (
            student_log_probs.exp() * (student_log_probs - mixture_log_probs)
        ).sum(dim=-1)
        reference = per_token[labels != -100].mean()
        observed = exact_divergence_vocab_chunked(
            student_chunked,
            teacher,
            labels,
            divergence="js",
            chunk_size=7,
        )
        reference.backward()
        observed.backward()

        torch.testing.assert_close(observed, reference, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(
            student_chunked.grad, student_reference.grad, rtol=1e-12, atol=1e-12
        )

    def test_recomputed_backward_matches_autograd_for_all_objectives(self):
        generator = torch.Generator().manual_seed(29)
        teacher = torch.randn(
            2, 3, 23, generator=generator, dtype=torch.float64
        )
        labels = torch.tensor(
            [[1, -100, 2], [3, 4, -100]], dtype=torch.long
        )
        for divergence in ("forward_kl", "reverse_kl", "js"):
            with self.subTest(divergence=divergence):
                student_reference = torch.randn(
                    2,
                    3,
                    23,
                    generator=generator,
                    dtype=torch.float64,
                    requires_grad=True,
                )
                student_recomputed = (
                    student_reference.detach().clone().requires_grad_(True)
                )
                reference = exact_divergence_vocab_chunked(
                    student_reference,
                    teacher,
                    labels,
                    divergence=divergence,
                    temperature=1.1,
                    chunk_size=7,
                )
                observed = recomputed_divergence_vocab_chunked(
                    student_recomputed,
                    teacher,
                    labels,
                    divergence=divergence,
                    temperature=1.1,
                    chunk_size=7,
                )
                reference.backward()
                observed.backward()

                torch.testing.assert_close(
                    observed, reference, rtol=1e-12, atol=1e-12
                )
                torch.testing.assert_close(
                    student_recomputed.grad,
                    student_reference.grad,
                    rtol=1e-11,
                    atol=1e-12,
                )

    def test_recomputed_js_saves_fewer_forward_tensors(self):
        generator = torch.Generator().manual_seed(31)
        teacher = torch.randn(1, 4, 257, generator=generator)
        labels = torch.ones(1, 4, dtype=torch.long)

        def saved_bytes(function):
            saved = []

            def pack(tensor):
                saved.append(tensor.numel() * tensor.element_size())
                return tensor

            student = torch.randn(
                1, 4, 257, generator=generator, requires_grad=True
            )
            with torch.autograd.graph.saved_tensors_hooks(
                pack, lambda tensor: tensor
            ):
                loss = function(student, teacher, labels)
            loss.backward()
            return sum(saved)

        ordinary = saved_bytes(
            lambda student, fixed_teacher, fixed_labels:
            exact_divergence_vocab_chunked(
                student,
                fixed_teacher,
                fixed_labels,
                divergence="js",
                chunk_size=64,
            )
        )
        recomputed = saved_bytes(
            lambda student, fixed_teacher, fixed_labels:
            recomputed_divergence_vocab_chunked(
                student,
                fixed_teacher,
                fixed_labels,
                divergence="js",
                chunk_size=64,
            )
        )

        self.assertLess(recomputed, ordinary)

    def test_recomputed_js_has_finite_bfloat16_student_only_gradient(self):
        student = torch.tensor(
            [[[2.0, 0.0, -1.0], [0.5, -0.5, 1.0]]],
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        teacher = torch.tensor(
            [[[-1.0, 0.0, 2.0], [1.0, -0.5, 0.5]]],
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        labels = torch.tensor([[1, -100]], dtype=torch.long)
        loss = recomputed_divergence_vocab_chunked(
            student,
            teacher,
            labels,
            divergence="js",
            temperature=1.1,
            chunk_size=2,
        )
        loss.backward()

        self.assertTrue(bool(torch.isfinite(student.grad).all()))
        self.assertGreater(float(student.grad.float().norm()), 0.0)
        self.assertIsNone(teacher.grad)

    def test_canonical_objectives_are_zero_at_equality_in_low_precision(self):
        logits = torch.tensor(
            [[[2.0, -1.0, 0.25], [0.5, 1.5, -3.0]]], dtype=torch.bfloat16
        )
        labels = torch.ones(1, 2, dtype=torch.long)

        for divergence in ("forward_kl", "reverse_kl", "js"):
            with self.subTest(divergence=divergence):
                observed = exact_divergence_vocab_chunked(
                    logits,
                    logits,
                    labels,
                    divergence=divergence,
                    chunk_size=2,
                )
                self.assertEqual(observed.dtype, torch.float32)
                self.assertEqual(float(observed), 0.0)

    def test_trainer_hook_uses_configured_canonical_objective_without_clipping(self):
        class FakeTrainer:
            pass

        fake_module = types.SimpleNamespace(OPSDTrainer=FakeTrainer)
        environment = {
            "OPSD_EXACT_JSD_VOCAB_CHUNK_SIZE": "2",
            "OPSD_TOKEN_DIVERGENCE": "js",
            "OPSD_RECOMPUTE_DIVERGENCE_BACKWARD": "1",
        }
        with mock.patch.dict(sys.modules, {"opsd_trainer": fake_module}), mock.patch.dict(
            os.environ, environment, clear=False
        ):
            launch_official_opsd._install_exact_jsd_chunking()
        self.assertEqual(FakeTrainer._opsd_token_divergence, "js")
        self.assertEqual(FakeTrainer._opsd_divergence_diagnostics_interval, 1)
        self.assertEqual(FakeTrainer._opsd_vocab_chunk_size, 2)
        self.assertIs(FakeTrainer._opsd_divergence_recompute, True)

        student = torch.tensor([[[2.0, 0.0, -1.0]]], dtype=torch.float64)
        teacher = torch.tensor([[[-1.0, 0.0, 2.0]]], dtype=torch.float64)
        labels = torch.ones(1, 1, dtype=torch.long)
        observed = FakeTrainer.generalized_jsd_loss(
            student,
            teacher,
            labels,
            beta=0.5,
            token_clip=0.05,
        )
        reference = exact_divergence_vocab_chunked(
            student, teacher, labels, divergence="js", chunk_size=2
        )
        torch.testing.assert_close(observed, reference)

    def test_divergence_statistics_report_masked_quantiles_and_entropies(self):
        student = torch.tensor(
            [[[2.0, 0.0, -1.0], [0.5, -0.5, 1.0]]], dtype=torch.float64
        )
        teacher = torch.tensor(
            [[[-1.0, 0.0, 2.0], [1.0, -0.5, 0.5]]], dtype=torch.float64
        )
        labels = torch.tensor([[1, -100]], dtype=torch.long)
        stats = divergence_statistics_vocab_chunked(
            student, teacher, labels, chunk_size=2
        )

        self.assertEqual(stats["token_count"], 1)
        for divergence in ("forward_kl", "reverse_kl", "js"):
            expected = exact_divergence_vocab_chunked(
                student,
                teacher,
                labels,
                divergence=divergence,
                chunk_size=2,
            )
            self.assertAlmostEqual(stats[divergence]["mean"], float(expected), places=12)
            self.assertEqual(stats[divergence]["nonfinite_count"], 0)
            self.assertEqual(stats[divergence]["negative_count"], 0)
            self.assertEqual(stats[divergence]["roundoff_negative_count"], 0)
            self.assertEqual(stats[divergence]["material_negative_count"], 0)
            self.assertEqual(stats[divergence]["p50"], stats[divergence]["mean"])
        self.assertGreater(stats["teacher_entropy"]["mean"], 0.0)
        self.assertGreater(stats["student_entropy"]["mean"], 0.0)

    def test_low_precision_roundoff_is_reported_but_not_material(self):
        generator = torch.Generator().manual_seed(0)
        teacher = (5 * torch.randn(1, 8, 8192, generator=generator)).to(
            torch.bfloat16
        )
        student = (
            teacher.float()
            + 0.01 * torch.randn(1, 8, 8192, generator=generator)
        ).to(torch.bfloat16)

        stats = divergence_statistics_vocab_chunked(
            student, teacher, chunk_size=2048
        )

        self.assertLess(stats["forward_kl"]["min"], -1e-7)
        self.assertGreater(stats["forward_kl"]["negative_count"], 0)
        self.assertGreater(stats["forward_kl"]["roundoff_negative_count"], 0)
        self.assertEqual(stats["forward_kl"]["material_negative_count"], 0)
        self.assertGreaterEqual(
            stats["forward_kl"]["roundoff_tolerance"],
            abs(stats["forward_kl"]["min"]),
        )
        self.assertAlmostEqual(
            canonical_negative_tolerance(torch.float32),
            64 * torch.finfo(torch.float32).eps,
        )

    def test_value_and_student_gradient_match_official_forward_kl(self):
        generator = torch.Generator().manual_seed(7)
        student_reference = torch.randn(
            2, 4, 17, generator=generator, dtype=torch.float64, requires_grad=True
        )
        student_chunked = student_reference.detach().clone().requires_grad_(True)
        teacher = torch.randn(2, 4, 17, generator=generator, dtype=torch.float64)
        labels = torch.tensor(
            [[-100, 1, 2, 3], [-100, -100, 4, 5]], dtype=torch.long
        )

        reference = official_beta_zero_loss(
            student_reference,
            teacher,
            labels,
            temperature=1.1,
            token_clip=0.05,
        )
        chunked = exact_forward_kl_vocab_chunked(
            student_chunked,
            teacher,
            labels,
            beta=0,
            temperature=1.1,
            token_clip=0.05,
            chunk_size=5,
        )
        reference.backward()
        chunked.backward()

        torch.testing.assert_close(chunked, reference, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(
            student_chunked.grad,
            student_reference.grad,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_unsupported_approximation_is_rejected(self):
        logits = torch.zeros(1, 1, 3)
        labels = torch.ones(1, 1, dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "top-k"):
            exact_forward_kl_vocab_chunked(
                logits,
                logits,
                labels,
                beta=0,
                top_k=2,
                chunk_size=2,
            )

    def test_low_precision_inputs_are_reduced_in_fp32(self):
        """Near-identical BF16 distributions must not acquire a negative KL."""
        generator = torch.Generator().manual_seed(11)
        teacher = torch.randn(1, 3, 257, generator=generator, dtype=torch.float32)
        # This is representative of an early LoRA update: distinguishable in
        # FP32, but vulnerable to cancellation if the KL reduction is BF16.
        student = (teacher + 0.003 * torch.randn(
            1, 3, 257, generator=generator, dtype=torch.float32
        )).to(torch.bfloat16).requires_grad_(True)
        labels = torch.ones(1, 3, dtype=torch.long)
        result = exact_forward_kl_vocab_chunked(
            student,
            teacher.to(torch.bfloat16),
            labels,
            beta=0,
            chunk_size=31,
        )
        self.assertEqual(result.dtype, torch.float32)
        self.assertGreaterEqual(float(result.detach()), 0.0)
        result.backward()
        self.assertTrue(torch.isfinite(student.grad).all())


if __name__ == "__main__":
    unittest.main()
