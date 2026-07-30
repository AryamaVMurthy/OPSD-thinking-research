"""Matched privileged-context views used by FiNOD."""

from __future__ import annotations

import torch


_FINAL_INSTRUCTION = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def _teacher_view(problem: str, auxiliary_context: str) -> str:
    return (
        f"Problem: {problem}\n\n"
        "The following auxiliary context may help with the reasoning. Treat it "
        "as contextual information, not as text to copy.\n"
        "=== Auxiliary Context Begin ===\n"
        f"{auxiliary_context}\n"
        "=== Auxiliary Context End ===\n\n"
        "Continue independently, check each inference, and revise the approach "
        "if the current path fails.\n"
        f"{_FINAL_INSTRUCTION}"
    )


def build_finod_teacher_views(
    *, problem: str, guide: str, answer: str
) -> dict[str, str]:
    """Return matched empty, procedural-guide, and destination-only views."""
    problem = str(problem).strip()
    guide = str(guide).strip()
    answer = str(answer).strip()
    if not problem or not guide or not answer:
        raise ValueError("FiNOD teacher views require problem, guide, and answer")
    return {
        "base": _teacher_view(
            problem,
            "No privileged auxiliary context is supplied for this view.",
        ),
        "guide": _teacher_view(problem, guide),
        "answer": _teacher_view(
            problem,
            "Destination-only control: the reported final answer is "
            f"\\boxed{{{answer}}}. No derivation is supplied.",
        ),
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


def attach_finod_teacher_views(collator, features, result: dict) -> None:
    """Replace the guide prompt and attach matched base/answer prompt tensors."""
    if collator.reason_first:
        raise RuntimeError("FiNOD requires reason_first=False")
    view_text = {"base": [], "guide": [], "answer": []}
    for feature in features:
        if "finod_answer_control" not in feature:
            raise ValueError("FiNOD row lacks finod_answer_control")
        views = build_finod_teacher_views(
            problem=feature["problem"],
            guide=feature["solution"],
            answer=feature["finod_answer_control"],
        )
        for name, user_text in views.items():
            rendered = collator.tokenizer.apply_chat_template(
                [{"role": "user", "content": user_text}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=collator.teacher_thinking,
            )
            view_text[name].append(rendered)

    encoded = {}
    for name in ("base", "guide", "answer"):
        encoded[name] = _encoded_view(
            collator.tokenizer, view_text[name], collator.max_length
        )

    guide, guide_lengths, guide_width = encoded["guide"]
    result.update(
        {
            "teacher_prompts": guide["input_ids"],
            "teacher_prompt_attention_mask": guide["attention_mask"],
            "teacher_prompt_length": guide_width,
            "teacher_prompt_lengths_per_example": torch.tensor(guide_lengths),
        }
    )
    for name in ("base", "answer"):
        tensors, lengths, width = encoded[name]
        result.update(
            {
                f"finod_{name}_prompts": tensors["input_ids"],
                f"finod_{name}_prompt_attention_mask": tensors[
                    "attention_mask"
                ],
                f"finod_{name}_prompt_length": width,
                f"finod_{name}_prompt_lengths_per_example": torch.tensor(
                    lengths
                ),
            }
        )


def install_finod_collator() -> None:
    """Install matched FiNOD views into upstream's structured collator."""
    from data_collator import SelfDistillationDataCollator

    original_call = SelfDistillationDataCollator.__call__
    if getattr(original_call, "_finod_wrapper", False):
        return

    def call_with_finod_views(self, features):
        result = original_call(self, features)
        attach_finod_teacher_views(self, features, result)
        return result

    call_with_finod_views._finod_wrapper = True
    SelfDistillationDataCollator.__call__ = call_with_finod_views
