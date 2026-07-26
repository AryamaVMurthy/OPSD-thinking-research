from __future__ import annotations

import unittest

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None

if torch is not None:
    from opsd_research.jsd import exact_forward_kl_vocab_chunked


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


if __name__ == "__main__":
    unittest.main()
