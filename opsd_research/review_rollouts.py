from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .records import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--per-class", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    records = [record for path in args.input for record in read_jsonl(path)]
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
