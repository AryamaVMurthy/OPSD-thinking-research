from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

from .config import MATH_DATASETS, load_config
from .generation_common import grade_math
from .records import (
    read_jsonl,
    validate_consistent_fields,
    validate_unique_complete,
)


def _bootstrap_problem_mean(values: dict[str, list[bool]], seed: int = 42) -> list[float]:
    rng = random.Random(seed)
    problem_ids = sorted(values)
    estimates = []
    for _ in range(5000):
        drawn = [rng.choice(problem_ids) for _ in problem_ids]
        estimates.append(
            mean(mean(int(value) for value in values[problem_id]) for problem_id in drawn)
        )
    return sorted(estimates)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--input", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = load_config(args.config).data
    records = [record for path in args.input for record in read_jsonl(path)]
    identity = validate_consistent_fields(
        records,
        (
            "model",
            "model_revision",
            "method",
            "checkpoint",
            "benchmark",
            "dataset_revision",
            "adapter_sha256",
            "seed_protocol",
        ),
    )
    expected_ids = [str(index) for index in range(MATH_DATASETS[config["dataset"]]["expected_count"])]
    observed_ids = sorted({str(record["problem_id"]) for record in records})
    # MathArena IDs are not guaranteed to be contiguous, so validate against observed
    # cardinality after the immutable dataset-count check in generation.
    if len(observed_ids) != len(expected_ids):
        raise ValueError(
            f"expected {len(expected_ids)} problem IDs, observed {len(observed_ids)}"
        )
    validate_unique_complete(records, observed_ids, config["samples_per_problem"])
    by_problem: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_problem[str(record["problem_id"])].append(record)

    correct_by_problem = {
        problem_id: [bool(record["correct"]) for record in problem_records]
        for problem_id, problem_records in by_problem.items()
    }
    avg = mean(
        int(record["correct"]) for problem_records in by_problem.values() for record in problem_records
    )
    pass_n = mean(any(values) for values in correct_by_problem.values())
    majority = []
    for problem_records in by_problem.values():
        formatted = [
            str(record["predicted_answer"])
            for record in problem_records
            if record.get("predicted_answer") is not None
        ]
        if not formatted:
            majority.append(False)
            continue
        answer = Counter(formatted).most_common(1)[0][0]
        majority.append(grade_math(answer, str(problem_records[0]["ground_truth"])))
    bootstrap = _bootstrap_problem_mean(correct_by_problem)
    lengths = sorted(int(record["output_tokens"]) for record in records)
    summary = {
        "schema_version": 1,
        "model": identity["model"],
        "model_revision": identity["model_revision"],
        "method": identity["method"],
        "checkpoint": identity["checkpoint"],
        "adapter_sha256": identity["adapter_sha256"],
        "seed_protocol": identity["seed_protocol"],
        "benchmark": identity["benchmark"],
        "dataset_revision": identity["dataset_revision"],
        "num_problems": len(by_problem),
        "samples_per_problem": config["samples_per_problem"],
        "avg_at_12": avg,
        "pass_at_12": pass_n,
        "maj_at_12": mean(majority),
        "bootstrap_95ci": [bootstrap[125], bootstrap[4874]],
        "format_rate": mean(bool(record["formatted"]) for record in records),
        "nonempty_thinking_rate": mean(
            bool(record["nonempty_thinking"]) for record in records
        ),
        "length_cutoff_rate": mean(
            record["finish_reason"] == "length" for record in records
        ),
        "output_tokens": {
            "mean": mean(lengths),
            "median": median(lengths),
            "p90": lengths[int(0.9 * (len(lengths) - 1))],
            "max": max(lengths),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
