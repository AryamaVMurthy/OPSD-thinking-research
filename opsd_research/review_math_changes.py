from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .records import read_jsonl
from .review_rollouts import apply_official_grades


def _pair_key(record: dict[str, Any]) -> tuple[str, int]:
    return str(record["problem_id"]), int(record["sample_index"])


def _render_rollout(label: str, record: dict[str, Any]) -> list[str]:
    return [
        f"### {label}",
        "",
        f"- Method: `{record['method']}`",
        f"- Checkpoint: `{record['checkpoint']}`",
        f"- Officially correct: `{bool(record['correct'])}`",
        f"- Predicted answer: `{record.get('predicted_answer')}`",
        f"- Output tokens: `{record.get('output_tokens')}`",
        f"- Finish reason: `{record.get('finish_reason')}`",
        "",
        "#### Reasoning",
        "",
        str(record.get("reasoning", "")),
        "",
        "#### Final answer",
        "",
        "```text",
        str(record.get("final", "")),
        "```",
        "",
    ]


def build_paired_review(
    baseline_records: list[dict[str, Any]],
    baseline_grades: list[dict[str, Any]],
    treatment_records: list[dict[str, Any]],
    treatment_grades: list[dict[str, Any]],
    *,
    per_class: int = 3,
    seed: int = 42,
) -> str:
    if per_class <= 0:
        raise ValueError("per_class must be positive")
    baseline = apply_official_grades(baseline_records, baseline_grades)
    treatment = apply_official_grades(treatment_records, treatment_grades)
    baseline_by_key = {_pair_key(record): record for record in baseline}
    treatment_by_key = {_pair_key(record): record for record in treatment}
    if len(baseline_by_key) != len(baseline):
        raise ValueError("duplicate baseline pair keys")
    if len(treatment_by_key) != len(treatment):
        raise ValueError("duplicate treatment pair keys")
    if baseline_by_key.keys() != treatment_by_key.keys():
        raise ValueError("baseline and treatment pair keys do not match")
    if not baseline_by_key:
        raise ValueError("cannot review empty runs")

    categories: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = (
        defaultdict(list)
    )
    for pair_key in sorted(baseline_by_key):
        base = baseline_by_key[pair_key]
        treated = treatment_by_key[pair_key]
        for field in ("model", "benchmark", "seed", "prompt_hash"):
            if base.get(field) != treated.get(field):
                raise ValueError(
                    f"pair {pair_key}: mismatched {field}: "
                    f"{base.get(field)!r} != {treated.get(field)!r}"
                )
        base_correct = bool(base["correct"])
        treated_correct = bool(treated["correct"])
        if base_correct and treated_correct:
            category = "both_correct"
        elif not base_correct and not treated_correct:
            category = "both_wrong"
        elif base_correct:
            category = "degraded"
        else:
            category = "improved"
        categories[category].append((base, treated))

    rng = random.Random(seed)
    category_order = ("degraded", "improved", "both_wrong", "both_correct")
    lines = [
        "# Paired math rollout review packet",
        "",
        "Official MathArena labels are hash-verified before pairing. Each pair",
        "uses the same problem, sample index, prompt hash, and random seed.",
        "",
    ]
    for category in category_order:
        pairs = categories[category]
        selected = rng.sample(pairs, min(per_class, len(pairs)))
        lines.extend(
            [
                f"## {category.replace('_', ' ').title()}",
                "",
                f"- Total paired samples in class: `{len(pairs)}`",
                f"- Selected for manual review: `{len(selected)}`",
                "",
            ]
        )
        for index, (base, treated) in enumerate(selected, start=1):
            lines.extend(
                [
                    f"### Pair {index}: problem {base['problem_id']} / sample {base['sample_index']}",
                    "",
                    f"- Ground truth: `{base.get('ground_truth')}`",
                    f"- Seed: `{base.get('seed')}`",
                    f"- Prompt SHA-256: `{base.get('prompt_hash')}`",
                    "",
                    "#### Problem",
                    "",
                    str(base.get("problem", "")),
                    "",
                ]
            )
            lines.extend(_render_rollout("Untouched baseline", base))
            lines.extend(_render_rollout("OPSD treatment", treated))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a deterministic paired math rollout review packet."
    )
    parser.add_argument("--baseline-input", required=True, nargs="+", type=Path)
    parser.add_argument("--baseline-grades", required=True, type=Path)
    parser.add_argument("--treatment-input", required=True, nargs="+", type=Path)
    parser.add_argument("--treatment-grades", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--per-class", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    packet = build_paired_review(
        [
            record
            for path in args.baseline_input
            for record in read_jsonl(path)
        ],
        read_jsonl(args.baseline_grades),
        [
            record
            for path in args.treatment_input
            for record in read_jsonl(path)
        ],
        read_jsonl(args.treatment_grades),
        per_class=args.per_class,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(packet + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output)}))


if __name__ == "__main__":
    main()
