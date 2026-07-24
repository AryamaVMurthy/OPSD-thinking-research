from __future__ import annotations

import re
from typing import Any


BOXED_PATTERN = re.compile(r"\\boxed\s*\{")


def extract_last_boxed(text: str) -> str | None:
    matches = list(BOXED_PATTERN.finditer(text))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    index = start
    while index < len(text):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index].strip()
        index += 1
    return None


def split_thinking(text: str) -> tuple[str, str, bool]:
    if "</think>" not in text:
        return "", text.strip(), False
    reasoning, final = text.rsplit("</think>", 1)
    reasoning = reasoning.replace("<think>", "", 1).strip()
    return reasoning, final.strip(), bool(reasoning)


def grade_math(predicted: str | None, ground_truth: str) -> bool:
    if predicted is None:
        return False
    try:
        from math_verify import parse, verify

        pred_parsed = parse(predicted)
        gt_parsed = parse(str(ground_truth))
        return bool(verify(gt_parsed, pred_parsed, timeout_seconds=5))
    except Exception:
        normalize = lambda value: (
            str(value).replace("$", "").replace(" ", "").lower().strip()
        )
        return normalize(predicted) == normalize(ground_truth)


def output_token_ids(output: Any) -> list[int]:
    ids = getattr(output, "token_ids", None)
    return list(ids) if ids is not None else []


def finish_reason(output: Any) -> str:
    value = getattr(output, "finish_reason", None)
    return "unknown" if value is None else str(value)
