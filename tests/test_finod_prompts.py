from __future__ import annotations

import torch

from opsd_research.finod_prompts import (
    attach_finod_teacher_views,
    build_finod_teacher_views,
)


def test_teacher_views_use_one_matched_wrapper_and_isolate_answer_content():
    views = build_finod_teacher_views(
        problem="Prove the requested claim.",
        guide="Split into the zero and nonzero cases.",
        answer="73",
    )

    assert set(views) == {"base", "guide", "answer"}
    assert "73" not in views["base"]
    assert "73" not in views["guide"]
    assert "\\boxed{73}" in views["answer"]
    for text in views.values():
        assert "=== Auxiliary Context Begin ===" in text
        assert "=== Auxiliary Context End ===" in text
        assert text.startswith("Problem: Prove the requested claim.")
        assert text.endswith(
            "Please reason step by step, and put your final answer within \\boxed{}."
        )


def test_collator_attachment_replaces_teacher_with_matched_guide_view():
    class FakeTokenizer:
        def __init__(self):
            self.rendered_batches = []

        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"]

        def __call__(
            self,
            prompts,
            *,
            padding,
            truncation,
            max_length,
            return_tensors=None,
        ):
            self.rendered_batches.append(list(prompts))
            lengths = [min(len(text), max_length) for text in prompts]
            if not return_tensors:
                return {"input_ids": [list(range(length)) for length in lengths]}
            width = max_length
            ids = torch.zeros(len(prompts), width, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for row, length in enumerate(lengths):
                ids[row, :length] = 1
                mask[row, :length] = 1
            return {"input_ids": ids, "attention_mask": mask}

    class FakeCollator:
        max_length = 4096
        teacher_thinking = True
        reason_first = False
        tokenizer = FakeTokenizer()

    result = {}
    attach_finod_teacher_views(
        FakeCollator(),
        [
            {
                "problem": "Problem A",
                "solution": "Use a parity split.",
                "finod_answer_control": "73",
            }
        ],
        result,
    )

    assert result["teacher_prompt_length"] > 0
    assert result["finod_base_prompt_length"] > 0
    assert result["finod_answer_prompt_length"] > 0
    rendered = [batch[0] for batch in FakeCollator.tokenizer.rendered_batches]
    guide_prompt = next(text for text in rendered if "Use a parity split." in text)
    answer_prompt = next(text for text in rendered if "\\boxed{73}" in text)
    assert "73" not in guide_prompt
    assert "Destination-only control" in answer_prompt
