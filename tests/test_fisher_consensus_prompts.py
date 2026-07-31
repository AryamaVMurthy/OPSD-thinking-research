from __future__ import annotations

import pytest
import torch

from opsd_research.fisher_consensus_prompts import (
    attach_fisher_consensus_views,
    build_fisher_consensus_views,
    decisive_plan_core,
)


def test_views_have_exact_deploy_anchor_and_matched_plan_wrappers():
    views = build_fisher_consensus_views(
        problem="Find the requested integer.",
        guides=["Factor symbolically.", "Use an invariant."],
        controls=["Introduce coordinates.", "Count complements."],
    )

    assert views["base"] == (
        "Problem: Find the requested integer.\n\n"
        "Please reason step by step, and put your final answer within \\boxed{}."
    )
    assert views["guides"][0].replace(
        "Factor symbolically.", "Introduce coordinates."
    ) == views["controls"][0]
    assert "correct" not in views["guides"][0].lower()
    assert "reported answer" not in views["guides"][0].lower()
    assert "reference" not in views["guides"][0].lower()


def test_decisive_plan_core_keeps_exactly_first_two_sentences():
    plan = (
        "Introduce coordinates. Derive the invariant from equal lengths. "
        "Verify every sign. If this route fails, try an alternative."
    )

    assert decisive_plan_core(plan, sentences=2) == (
        "Introduce coordinates. Derive the invariant from equal lengths."
    )


def test_decisive_core_is_applied_symmetrically_to_guides_and_controls():
    views = build_fisher_consensus_views(
        problem="Find the requested integer.",
        guides=[
            "Factor the polynomial. Compare multiplicities. Use a fallback.",
            "Introduce a recurrence. Solve its fixed point. Check another route.",
        ],
        controls=[
            "Draw the altitude. Apply similarity. Verify with coordinates.",
            "Count complements. Use inclusion-exclusion. Try an alternative.",
        ],
        plan_core_sentences=2,
    )

    rendered = "\n".join([*views["guides"], *views["controls"]]).lower()
    assert "factor the polynomial. compare multiplicities." in rendered
    assert "draw the altitude. apply similarity." in rendered
    assert "fallback" not in rendered
    assert "verify" not in rendered
    assert "alternative" not in rendered


def test_decisive_core_rejects_short_or_duplicate_plan_cores():
    with pytest.raises(ValueError, match="at least 2 sentences"):
        decisive_plan_core("Only one sentence.", sentences=2)

    with pytest.raises(ValueError, match="distinct"):
        build_fisher_consensus_views(
            problem="Find the requested integer.",
            guides=[
                "Use symmetry. Count orbits. First fallback.",
                "Use symmetry. Count orbits. Second fallback.",
            ],
            controls=[
                "Factor first. Compare roots. Check signs.",
                "Use coordinates. Eliminate a variable. Verify.",
            ],
            plan_core_sentences=2,
        )

    with pytest.raises(ValueError, match="answer claim"):
        decisive_plan_core(
            "Factor the expression. The final answer is boxed here.",
            sentences=2,
        )


def test_collator_attaches_all_pairs_without_answer_view():
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
            ids = torch.zeros(len(prompts), max_length, dtype=torch.long)
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
    attach_fisher_consensus_views(
        FakeCollator(),
        [
            {
                "problem": "Problem A",
                "solution": "guide zero",
                "fisher_guides": ["guide zero", "guide one", "guide two"],
                "fisher_controls": [
                    "control zero",
                    "control one",
                    "control two",
                ],
            }
        ],
        result,
    )

    assert result["fisher_pair_count"] == 3
    assert result["teacher_prompt_length"] > 0
    assert result["fisher_base_prompt_length"] > 0
    for pair in range(3):
        assert result[f"fisher_guide_{pair}_prompt_length"] > 0
        assert result[f"fisher_control_{pair}_prompt_length"] > 0
    assert not any("answer" in key for key in result)
    assert not any("reference" in key for key in result)
