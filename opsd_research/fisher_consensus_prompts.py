"""Matched prompt views for answer-free Fisher consensus distillation."""

from __future__ import annotations

import re
from collections.abc import Sequence

import torch

from .fisher_guidance import validate_answer_free_procedure


_FINAL_INSTRUCTION = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def decisive_plan_core(plan: str, *, sentences: int) -> str:
    """Return exactly the first complete procedural sentences of a plan."""
    if (
        not isinstance(sentences, int)
        or isinstance(sentences, bool)
        or sentences < 1
    ):
        raise ValueError("plan core sentences must be a positive integer")
    parts = [
        part.strip()
        for part in _SENTENCE_BOUNDARY.split(str(plan).strip())
        if part.strip()
    ]
    if len(parts) < sentences:
        raise ValueError(
            f"decisive plan core requires at least {sentences} sentences"
        )
    core = " ".join(parts[:sentences])
    validate_answer_free_procedure(core)
    return core


def _plan_view(problem: str, plan: str) -> str:
    return (
        f"Problem: {problem}\n\n"
        "A tentative procedural plan is provided below. Use a step only when "
        "it is locally supported by the problem; otherwise continue "
        "independently.\n"
        "=== Procedural Plan Begin ===\n"
        f"{plan}\n"
        "=== Procedural Plan End ===\n\n"
        f"{_FINAL_INSTRUCTION}"
    )


def build_fisher_consensus_views(
    *,
    problem: str,
    guides: Sequence[str],
    controls: Sequence[str],
    plan_core_sentences: int | None = None,
) -> dict[str, object]:
    """Build the deploy anchor and exactly matched positive/control views."""
    problem = str(problem).strip()
    guides = [str(plan).strip() for plan in guides]
    controls = [str(plan).strip() for plan in controls]
    if not problem:
        raise ValueError("Fisher consensus views require a problem")
    if len(guides) < 2 or len(guides) != len(controls):
        raise ValueError("Fisher consensus views require matched plan pairs")
    if any(not plan for plan in [*guides, *controls]):
        raise ValueError("Fisher consensus views require nonempty plans")
    if plan_core_sentences is not None:
        guides = [
            decisive_plan_core(plan, sentences=plan_core_sentences)
            for plan in guides
        ]
        controls = [
            decisive_plan_core(plan, sentences=plan_core_sentences)
            for plan in controls
        ]
        normalized = [" ".join(plan.lower().split()) for plan in guides]
        if len(set(normalized)) != len(normalized):
            raise ValueError("decisive guide plan cores must be distinct")
        normalized = [" ".join(plan.lower().split()) for plan in controls]
        if len(set(normalized)) != len(normalized):
            raise ValueError("decisive control plan cores must be distinct")
    return {
        "base": f"Problem: {problem}\n\n{_FINAL_INSTRUCTION}",
        "guides": [_plan_view(problem, plan) for plan in guides],
        "controls": [_plan_view(problem, plan) for plan in controls],
    }


def _encoded_view(tokenizer, prompts: list[str], max_length: int):
    unpadded = tokenizer(
        prompts,
        padding=False,
        truncation=True,
        max_length=max_length,
    )
    lengths = [len(ids) for ids in unpadded["input_ids"]]
    width = max(lengths)
    encoded = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=width,
        return_tensors="pt",
    )
    return encoded, lengths, width


def _attach_encoded(result: dict, name: str, encoded) -> None:
    tensors, lengths, width = encoded
    result.update(
        {
            f"{name}_prompts": tensors["input_ids"],
            f"{name}_prompt_attention_mask": tensors["attention_mask"],
            f"{name}_prompt_length": width,
            f"{name}_prompt_lengths_per_example": torch.tensor(lengths),
        }
    )


def attach_fisher_consensus_views(
    collator,
    features,
    result: dict,
    *,
    plan_core_sentences: int | None = None,
) -> None:
    """Attach K positive/control views without constructing an answer view."""
    if collator.reason_first:
        raise RuntimeError("Fisher consensus requires reason_first=False")
    all_views = [
        build_fisher_consensus_views(
            problem=feature["problem"],
            guides=feature["fisher_guides"],
            controls=feature["fisher_controls"],
            plan_core_sentences=plan_core_sentences,
        )
        for feature in features
    ]
    pair_counts = {len(views["guides"]) for views in all_views}
    if len(pair_counts) != 1:
        raise ValueError("Fisher consensus batch has inconsistent pair counts")
    pair_count = next(iter(pair_counts))

    def render(user_text: str) -> str:
        return collator.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=collator.teacher_thinking,
        )

    base = _encoded_view(
        collator.tokenizer,
        [render(str(views["base"])) for views in all_views],
        collator.max_length,
    )
    _attach_encoded(result, "fisher_base", base)

    for pair in range(pair_count):
        for kind in ("guide", "control"):
            plural = f"{kind}s"
            encoded = _encoded_view(
                collator.tokenizer,
                [
                    render(str(views[plural][pair]))
                    for views in all_views
                ],
                collator.max_length,
            )
            _attach_encoded(result, f"fisher_{kind}_{pair}", encoded)

    # Upstream only needs one teacher prompt to append the on-policy student
    # completion.  The loss hook reads every independently encoded pair.
    teacher = _encoded_view(
        collator.tokenizer,
        [render(str(views["guides"][0])) for views in all_views],
        collator.max_length,
    )
    tensors, lengths, width = teacher
    result.update(
        {
            "teacher_prompts": tensors["input_ids"],
            "teacher_prompt_attention_mask": tensors["attention_mask"],
            "teacher_prompt_length": width,
            "teacher_prompt_lengths_per_example": torch.tensor(lengths),
            "fisher_pair_count": pair_count,
        }
    )


def install_fisher_consensus_collator(
    *, plan_core_sentences: int | None = None
) -> None:
    """Install answer-free plan views into the upstream collator."""
    from data_collator import SelfDistillationDataCollator

    original_call = SelfDistillationDataCollator.__call__
    if getattr(original_call, "_fisher_consensus_wrapper", False):
        return

    def call_with_fisher_consensus_views(self, features):
        result = original_call(self, features)
        attach_fisher_consensus_views(
            self,
            features,
            result,
            plan_core_sentences=plan_core_sentences,
        )
        return result

    call_with_fisher_consensus_views._fisher_consensus_wrapper = True
    SelfDistillationDataCollator.__call__ = (
        call_with_fisher_consensus_views
    )
