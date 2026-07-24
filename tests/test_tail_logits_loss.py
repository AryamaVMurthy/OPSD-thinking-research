from __future__ import annotations

import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from opsd_research.tail_logits_loss import _generation_logits

try:
    from transformers import Qwen3Config, Qwen3ForCausalLM
except (ImportError, ModuleNotFoundError):
    Qwen3Config = None
    Qwen3ForCausalLM = None


@unittest.skipIf(torch is None, "torch is available in the pinned Turing environment")
class TailLogitsTests(unittest.TestCase):
    def test_drops_only_the_unscored_last_position(self):
        outputs = type(
            "Outputs",
            (),
            {"logits": torch.arange(2 * 5 * 3).reshape(2, 5, 3)},
        )()

        result = _generation_logits(outputs, generation_length=4)

        torch.testing.assert_close(result, outputs.logits[:, :-1, :])

    def test_unexpected_model_output_is_rejected(self):
        outputs = type("Outputs", (), {"logits": torch.zeros(1, 4, 2)})()
        with self.assertRaisesRegex(RuntimeError, "expected 5"):
            _generation_logits(outputs, generation_length=4)

    @unittest.skipIf(
        Qwen3ForCausalLM is None,
        "transformers with Qwen3 is available in the pinned Turing environment",
    )
    def test_qwen3_tail_logits_match_upstream_full_slices_and_gradients(self):
        torch.manual_seed(42)
        model = Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=101,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                max_position_embeddings=128,
            )
        )
        model.eval()
        generation_length = 7

        for prompt_length in (5, 11):
            input_ids = torch.randint(
                0, model.config.vocab_size, (2, prompt_length + generation_length)
            )
            attention_mask = torch.ones_like(input_ids)

            full = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits[:, prompt_length - 1 : -1, :]
            tail = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                logits_to_keep=generation_length + 1,
            ).logits[:, :-1, :]

            torch.testing.assert_close(tail, full, rtol=0, atol=0)

            model.zero_grad(set_to_none=True)
            full.float().square().mean().backward()
            full_grad = model.lm_head.weight.grad.detach().clone()
            model.zero_grad(set_to_none=True)
            tail.float().square().mean().backward()
            tail_grad = model.lm_head.weight.grad.detach().clone()
            torch.testing.assert_close(tail_grad, full_grad, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
