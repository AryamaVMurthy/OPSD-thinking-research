from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from .records import read_jsonl


def _review_key(record: dict) -> tuple[str, str, str, str, str, int]:
    return (
        str(record["model"]),
        str(record["method"]),
        str(record["checkpoint"]),
        str(record["benchmark"]),
        str(record["problem_id"]),
        int(record["sample_index"]),
    )


def apply_official_grades(
    records: list[dict],
    sidecars: list[dict],
) -> list[dict]:
    """Overlay verified official labels on copies used only for review."""
    raw_by_key = {_review_key(record): record for record in records}
    grade_by_key = {_review_key(sidecar): sidecar for sidecar in sidecars}
    if len(raw_by_key) != len(records):
        raise ValueError("duplicate raw rollout keys")
    if len(grade_by_key) != len(sidecars):
        raise ValueError("duplicate official grade keys")
    if raw_by_key.keys() != grade_by_key.keys():
        raise ValueError("raw rollout and official grade keys do not match")

    merged = []
    for record in records:
        sidecar = grade_by_key[_review_key(record)]
        response_hash = hashlib.sha256(
            str(record["response"]).encode("utf-8")
        ).hexdigest()
        if sidecar["response_sha256"] != response_hash:
            raise ValueError(
                f"response hash mismatch for {_review_key(record)}"
            )
        reviewed = dict(record)
        reviewed["legacy_correct"] = bool(record["correct"])
        reviewed["legacy_predicted_answer"] = record.get("predicted_answer")
        reviewed["correct"] = bool(sidecar["official_correct"])
        reviewed["predicted_answer"] = sidecar["official_predicted_answer"]
        reviewed["review_grading_protocol"] = "official-matharena-parser-v1"
        merged.append(reviewed)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--per-class", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--official-grades", type=Path)
    args = parser.parse_args()
    records = [record for path in args.input for record in read_jsonl(path)]
    if args.official_grades:
        records = apply_official_grades(
            records,
            read_jsonl(args.official_grades),
        )
    if not all("correct" in record for record in records):
        raise ValueError("review requires scored records with a correct field")
    rng = random.Random(args.seed)
    correct = [record for record in records if record["correct"]]
    incorrect = [record for record in records if not record["correct"]]
    selected = [
        ("correct", record)
        for record in rng.sample(correct, min(args.per_class, len(correct)))
    ] + [
        ("incorrect", record)
        for record in rng.sample(incorrect, min(args.per_class, len(incorrect)))
    ]
    lines = ["# Rollout review packet", ""]
    for index, (classification, record) in enumerate(selected, start=1):
        lines.extend(
            [
                f"## {index}. {classification.upper()} — {record['problem_id']} / sample {record['sample_index']}",
                "",
                f"- Model: `{record['model']}`",
                f"- Method: `{record['method']}`",
                f"- Checkpoint: `{record['checkpoint']}`",
                f"- Tokens: `{record.get('output_tokens')}`",
                f"- Finish reason: `{record.get('finish_reason')}`",
                f"- Ground truth: `{record.get('ground_truth', 'execution tests')}`",
                f"- Predicted answer: `{record.get('predicted_answer', 'see code')}`",
                "",
                "### Problem",
                "",
                str(record["problem"]),
                "",
                "### Reasoning",
                "",
                str(record.get("reasoning", "")),
                "",
                "### Final/code",
                "",
                "```text",
                str(record.get("final", record.get("code", ""))),
                "```",
                "",
            ]
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"reviewed": len(selected), "output": str(args.output)}))


if __name__ == "__main__":
    main()
