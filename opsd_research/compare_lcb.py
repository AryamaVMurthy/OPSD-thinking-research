from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from .records import read_jsonl


def _pair_key(record: dict[str, Any]) -> tuple[str, int]:
    return str(record["problem_id"]), int(record["sample_index"])


def _pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    if not 1 <= k <= num_samples:
        raise ValueError(f"k must be in [1, {num_samples}], found {k}")
    if num_samples - num_correct < k:
        return 1.0
    return 1.0 - math.comb(num_samples - num_correct, k) / math.comb(
        num_samples, k
    )


def _problem_metrics(
    records: list[dict[str, Any]],
    *,
    samples_per_problem: int,
    k_values: tuple[int, ...],
) -> dict[str, dict[int, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in sorted(records, key=_pair_key):
        grouped[str(record["problem_id"])].append(record)
    result = {}
    for problem_id, rows in grouped.items():
        sample_indices = {int(row["sample_index"]) for row in rows}
        if (
            len(rows) != samples_per_problem
            or sample_indices != set(range(samples_per_problem))
        ):
            raise ValueError(
                f"problem {problem_id}: expected sample indices "
                f"0..{samples_per_problem - 1}, found {sorted(sample_indices)}"
            )
        num_correct = sum(bool(row["correct"]) for row in rows)
        result[problem_id] = {
            k: _pass_at_k(samples_per_problem, num_correct, k)
            for k in k_values
        }
    return result


def _metric_triplet(
    baseline: dict[str, dict[int, float]],
    treatment: dict[str, dict[int, float]],
    k: int,
) -> dict[str, float]:
    problem_ids = sorted(baseline)
    baseline_value = mean(baseline[problem_id][k] for problem_id in problem_ids)
    treatment_value = mean(
        treatment[problem_id][k] for problem_id in problem_ids
    )
    return {
        "baseline": baseline_value,
        "treatment": treatment_value,
        "delta": treatment_value - baseline_value,
    }


def _paired_bootstrap(
    baseline: dict[str, dict[int, float]],
    treatment: dict[str, dict[int, float]],
    *,
    k_values: tuple[int, ...],
    bootstrap_samples: int,
    seed: int = 42,
) -> dict[int, list[float]]:
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    rng = random.Random(seed)
    problem_ids = sorted(baseline)
    estimates = {k: [] for k in k_values}
    for _ in range(bootstrap_samples):
        drawn = [rng.choice(problem_ids) for _ in problem_ids]
        for k in k_values:
            estimates[k].append(
                mean(
                    treatment[problem_id][k] - baseline[problem_id][k]
                    for problem_id in drawn
                )
            )
    lower = int(0.025 * bootstrap_samples)
    upper = max(lower, int(0.975 * bootstrap_samples) - 1)
    return {
        k: [sorted(values)[lower], sorted(values)[upper]]
        for k, values in estimates.items()
    }


def compare_paired_lcb(
    baseline_records: list[dict[str, Any]],
    treatment_records: list[dict[str, Any]],
    *,
    samples_per_problem: int,
    k_values: Iterable[int] = (1, 5),
    bootstrap_samples: int = 5000,
) -> dict[str, Any]:
    """Compare paired officially executed LiveCodeBench generations."""
    k_values = tuple(k_values)
    baseline_by_key = {_pair_key(record): record for record in baseline_records}
    treatment_by_key = {_pair_key(record): record for record in treatment_records}
    if len(baseline_by_key) != len(baseline_records):
        raise ValueError("duplicate baseline generation keys")
    if len(treatment_by_key) != len(treatment_records):
        raise ValueError("duplicate treatment generation keys")
    if baseline_by_key.keys() != treatment_by_key.keys():
        raise ValueError("baseline and treatment generation keys do not match")
    if not baseline_by_key:
        raise ValueError("cannot compare empty runs")

    for record_key, baseline_record in baseline_by_key.items():
        treatment_record = treatment_by_key[record_key]
        for field in (
            "model",
            "model_revision",
            "benchmark",
            "dataset_revision",
            "seed_protocol",
            "seed",
            "prompt_hash",
        ):
            if baseline_record.get(field) != treatment_record.get(field):
                raise ValueError(
                    f"pair {record_key}: mismatched {field}: "
                    f"{baseline_record.get(field)!r} != "
                    f"{treatment_record.get(field)!r}"
                )

    baseline_problem = _problem_metrics(
        baseline_records,
        samples_per_problem=samples_per_problem,
        k_values=k_values,
    )
    treatment_problem = _problem_metrics(
        treatment_records,
        samples_per_problem=samples_per_problem,
        k_values=k_values,
    )
    if baseline_problem.keys() != treatment_problem.keys():
        raise ValueError("baseline and treatment problem IDs do not match")

    changes = Counter()
    for record_key, baseline_record in baseline_by_key.items():
        baseline_correct = bool(baseline_record["correct"])
        treatment_correct = bool(treatment_by_key[record_key]["correct"])
        if baseline_correct and treatment_correct:
            changes["both_correct"] += 1
        elif not baseline_correct and not treatment_correct:
            changes["both_wrong"] += 1
        elif baseline_correct:
            changes["degraded"] += 1
        else:
            changes["improved"] += 1

    intervals = _paired_bootstrap(
        baseline_problem,
        treatment_problem,
        k_values=k_values,
        bootstrap_samples=bootstrap_samples,
    )
    first_baseline = baseline_records[0]
    first_treatment = treatment_records[0]
    result: dict[str, Any] = {
        "schema_version": 1,
        "comparison_protocol": "paired-problem-cluster-bootstrap-v1",
        "pairing_verified": True,
        "model": first_baseline["model"],
        "benchmark": first_baseline["benchmark"],
        "baseline_method": first_baseline["method"],
        "baseline_checkpoint": first_baseline["checkpoint"],
        "treatment_method": first_treatment["method"],
        "treatment_checkpoint": first_treatment["checkpoint"],
        "num_problems": len(baseline_problem),
        "samples_per_problem": samples_per_problem,
        "delta_bootstrap_95ci": {
            f"pass_at_{k}": intervals[k] for k in k_values
        },
        "paired_sample_changes": {
            key: changes[key]
            for key in ("both_correct", "both_wrong", "degraded", "improved")
        },
    }
    result.update(
        {
            f"pass_at_{k}": _metric_triplet(
                baseline_problem, treatment_problem, k
            )
            for k in k_values
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two paired, officially executed LiveCodeBench runs."
    )
    parser.add_argument("--baseline-graded", required=True, type=Path)
    parser.add_argument("--treatment-graded", required=True, type=Path)
    parser.add_argument("--samples-per-problem", required=True, type=int)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    comparison = compare_paired_lcb(
        read_jsonl(args.baseline_graded),
        read_jsonl(args.treatment_graded),
        samples_per_problem=args.samples_per_problem,
        bootstrap_samples=args.bootstrap_samples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
