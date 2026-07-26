"""Summarize CH cache coverage, cost, diversity, and attempt marginal value."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from .generation_common import extract_last_boxed, grade_math
from .records import read_jsonl


def summarize_ch_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot summarize an empty CH cache")
    accepted = [record for record in records if bool(record.get("accepted"))]
    if not accepted:
        raise ValueError("CH cache has no accepted records")
    attempt_counts = {int(record["blind_attempt_count"]) for record in accepted}
    if len(attempt_counts) != 1:
        raise ValueError("accepted CH records use inconsistent attempt counts")
    num_attempts = attempt_counts.pop()
    if num_attempts not in {2, 3}:
        raise ValueError("CH summary requires two or three attempts")

    correct_by_attempt = [0] * num_attempts
    formatted_by_attempt = [0] * num_attempts
    output_tokens_by_attempt: list[list[int]] = [[] for _ in range(num_attempts)]
    audit_tokens: list[int] = []
    teacher_dossier_tokens: list[int] = []
    teacher_prompt_tokens: list[int] = []
    any_correct = 0
    mixed_correctness = 0
    unique_recoveries = 0
    answer_diversity: list[int] = []
    for record in accepted:
        attempts = list(record.get("blind_attempts", []))
        tokens = list(record.get("blind_attempt_output_tokens", []))
        if len(attempts) != num_attempts or len(tokens) != num_attempts:
            raise ValueError("accepted CH record lacks complete attempt diagnostics")
        reference = str(record["reference_answer"])
        predictions = [extract_last_boxed(str(attempt)) for attempt in attempts]
        correctness = [
            bool(grade_math(prediction, reference)) if prediction is not None else False
            for prediction in predictions
        ]
        for index, (prediction, correct, token_count) in enumerate(
            zip(predictions, correctness, tokens, strict=True)
        ):
            formatted_by_attempt[index] += int(prediction is not None)
            correct_by_attempt[index] += int(correct)
            output_tokens_by_attempt[index].append(int(token_count))
        any_correct += int(any(correctness))
        mixed_correctness += int(any(correctness) and not all(correctness))
        if num_attempts == 3:
            unique_recoveries += int(
                correctness[2] and not correctness[0] and not correctness[1]
            )
        answer_diversity.append(len({answer for answer in predictions if answer is not None}))
        audit_tokens.append(int(record.get("audit_output_tokens", 0)))
        dossier_tokens = int(record.get("teacher_dossier_tokens", 0))
        if dossier_tokens < 1:
            raise ValueError("accepted CH record lacks a teacher-dossier token count")
        teacher_dossier_tokens.append(dossier_tokens)
        prompt_tokens = int(record.get("teacher_prompt_tokens", 0))
        if prompt_tokens < 1:
            raise ValueError("accepted CH record lacks a teacher-prompt token count")
        teacher_prompt_tokens.append(prompt_tokens)

    count = len(accepted)
    rejection_reasons = Counter(
        str(record.get("rejection_reason", "unspecified"))
        for record in records
        if not bool(record.get("accepted"))
    )
    total_blind_tokens = sum(sum(values) for values in output_tokens_by_attempt)
    total_audit_tokens = sum(audit_tokens)
    return {
        "schema_version": 1,
        "requested_examples": len(records),
        "accepted_examples": count,
        "rejected_examples": len(records) - count,
        "acceptance_rate": count / len(records),
        "blind_attempts_per_problem": num_attempts,
        "formatted_rate_by_attempt": [
            value / count for value in formatted_by_attempt
        ],
        "correct_rate_by_attempt": [value / count for value in correct_by_attempt],
        "any_attempt_correct_rate": any_correct / count,
        "mixed_correctness_rate": mixed_correctness / count,
        "mean_distinct_boxed_answers": mean(answer_diversity),
        "third_attempt_unique_recoveries": unique_recoveries,
        "third_attempt_unique_recovery_rate": unique_recoveries / count,
        "mean_output_tokens_by_attempt": [
            mean(values) for values in output_tokens_by_attempt
        ],
        "mean_audit_output_tokens": mean(audit_tokens),
        "mean_teacher_dossier_tokens": mean(teacher_dossier_tokens),
        "max_teacher_dossier_tokens": max(teacher_dossier_tokens),
        "teacher_dossiers_over_12000": sum(
            tokens > 12_000 for tokens in teacher_dossier_tokens
        ),
        "mean_teacher_prompt_tokens": mean(teacher_prompt_tokens),
        "max_teacher_prompt_tokens": max(teacher_prompt_tokens),
        "teacher_prompts_over_24576": sum(
            tokens > 24_576 for tokens in teacher_prompt_tokens
        ),
        "total_blind_generation_tokens": total_blind_tokens,
        "total_audit_generation_tokens": total_audit_tokens,
        "total_generation_tokens": total_blind_tokens + total_audit_tokens,
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        summary = summarize_ch_records(read_jsonl(args.input))
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
