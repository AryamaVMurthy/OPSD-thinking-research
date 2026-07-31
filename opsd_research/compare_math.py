from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from statistics import mean
from pathlib import Path
from typing import Any

from .records import read_jsonl
from .review_rollouts import apply_official_grades


def _pair_key(record: dict[str, Any]) -> tuple[str, int]:
    return str(record["problem_id"]), int(record["sample_index"])


def _problem_metrics(
    records: list[dict[str, Any]],
    *,
    samples_per_problem: int,
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in sorted(records, key=_pair_key):
        grouped[str(record["problem_id"])].append(record)
    result = {}
    for problem_id, rows in grouped.items():
        if len(rows) != samples_per_problem:
            raise ValueError(
                f"problem {problem_id}: expected {samples_per_problem} samples, "
                f"found {len(rows)}"
            )
        predictions = [
            str(row["predicted_answer"])
            for row in rows
            if row.get("predicted_answer") is not None
        ]
        majority_correct = False
        if predictions:
            counts = Counter(predictions)
            highest = max(counts.values())
            winner = next(value for value in predictions if counts[value] == highest)
            majority_correct = next(
                bool(row["correct"])
                for row in rows
                if str(row.get("predicted_answer")) == winner
            )
        result[problem_id] = {
            "avg": mean(int(bool(row["correct"])) for row in rows),
            "pass": float(any(bool(row["correct"]) for row in rows)),
            "maj": float(majority_correct),
        }
    return result


def _metric_triplet(
    baseline: dict[str, dict[str, float]],
    treatment: dict[str, dict[str, float]],
    metric: str,
) -> dict[str, float]:
    problem_ids = sorted(baseline)
    base_value = mean(baseline[problem_id][metric] for problem_id in problem_ids)
    treatment_value = mean(
        treatment[problem_id][metric] for problem_id in problem_ids
    )
    return {
        "baseline": base_value,
        "treatment": treatment_value,
        "delta": treatment_value - base_value,
    }


def _output_tokens(record: dict[str, Any]) -> int:
    value = record.get("output_tokens")
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("paired generation record has invalid output_tokens")
    return value


def _is_cutoff(record: dict[str, Any]) -> bool:
    if record.get("finish_reason") == "length":
        return True
    limit = record.get("effective_max_new_tokens")
    return (
        isinstance(limit, int)
        and not isinstance(limit, bool)
        and limit > 0
        and _output_tokens(record) >= limit
    )


def _generation_diagnostics(
    baseline: dict[tuple[str, int], dict[str, Any]],
    treatment: dict[tuple[str, int], dict[str, Any]],
    transitions: dict[
        str,
        list[tuple[dict[str, Any], dict[str, Any]]],
    ],
) -> tuple[dict[str, float | int], dict[str, dict[str, float | int | None]]]:
    baseline_rows = list(baseline.values())
    treatment_rows = [treatment[key] for key in baseline]
    baseline_cutoffs = sum(_is_cutoff(row) for row in baseline_rows)
    treatment_cutoffs = sum(_is_cutoff(row) for row in treatment_rows)
    aggregate: dict[str, float | int] = {
        "baseline_mean_output_tokens": mean(
            _output_tokens(row) for row in baseline_rows
        ),
        "treatment_mean_output_tokens": mean(
            _output_tokens(row) for row in treatment_rows
        ),
        "mean_output_token_delta": mean(
            _output_tokens(treatment[key]) - _output_tokens(baseline[key])
            for key in baseline
        ),
        "baseline_cutoffs": baseline_cutoffs,
        "treatment_cutoffs": treatment_cutoffs,
        "cutoff_delta": treatment_cutoffs - baseline_cutoffs,
    }
    localized = {}
    for name in ("both_correct", "both_wrong", "degraded", "improved"):
        rows = transitions.get(name, [])
        base_count = sum(_is_cutoff(base) for base, _ in rows)
        treatment_count = sum(
            _is_cutoff(candidate) for _, candidate in rows
        )
        localized[name] = {
            "count": len(rows),
            "mean_output_token_delta": (
                mean(
                    _output_tokens(candidate) - _output_tokens(base)
                    for base, candidate in rows
                )
                if rows
                else None
            ),
            "baseline_cutoffs": base_count,
            "treatment_cutoffs": treatment_count,
            "cutoff_delta": treatment_count - base_count,
        }
    return aggregate, localized


def _paired_bootstrap(
    baseline: dict[str, dict[str, float]],
    treatment: dict[str, dict[str, float]],
    *,
    bootstrap_samples: int,
    seed: int = 42,
) -> dict[str, list[float]]:
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    rng = random.Random(seed)
    problem_ids = sorted(baseline)
    estimates = {metric: [] for metric in ("avg", "pass", "maj")}
    for _ in range(bootstrap_samples):
        drawn = [rng.choice(problem_ids) for _ in problem_ids]
        for metric in estimates:
            estimates[metric].append(
                mean(
                    treatment[problem_id][metric]
                    - baseline[problem_id][metric]
                    for problem_id in drawn
                )
            )
    lower = int(0.025 * bootstrap_samples)
    upper = max(lower, int(0.975 * bootstrap_samples) - 1)
    return {
        metric: [sorted(values)[lower], sorted(values)[upper]]
        for metric, values in estimates.items()
    }


def compare_paired_math(
    baseline_records: list[dict[str, Any]],
    baseline_grades: list[dict[str, Any]],
    treatment_records: list[dict[str, Any]],
    treatment_grades: list[dict[str, Any]],
    *,
    samples_per_problem: int,
    bootstrap_samples: int = 5000,
    take_first_samples: bool = False,
) -> dict[str, Any]:
    """Compare two official-scored math runs using paired problem clusters."""
    if take_first_samples:
        baseline_records = [
            record
            for record in baseline_records
            if int(record["sample_index"]) < samples_per_problem
        ]
        baseline_grades = [
            grade
            for grade in baseline_grades
            if int(grade["sample_index"]) < samples_per_problem
        ]
        treatment_records = [
            record
            for record in treatment_records
            if int(record["sample_index"]) < samples_per_problem
        ]
        treatment_grades = [
            grade
            for grade in treatment_grades
            if int(grade["sample_index"]) < samples_per_problem
        ]
    baseline = apply_official_grades(baseline_records, baseline_grades)
    treatment = apply_official_grades(treatment_records, treatment_grades)
    baseline_by_key = {_pair_key(record): record for record in baseline}
    treatment_by_key = {_pair_key(record): record for record in treatment}
    if baseline_by_key.keys() != treatment_by_key.keys():
        raise ValueError("baseline and treatment generation keys do not match")
    if not baseline_by_key:
        raise ValueError("cannot compare empty runs")

    for record_key, base_record in baseline_by_key.items():
        treatment_record = treatment_by_key[record_key]
        for field in ("model", "benchmark", "seed", "prompt_hash"):
            if base_record.get(field) != treatment_record.get(field):
                raise ValueError(
                    f"pair {record_key}: mismatched {field}: "
                    f"{base_record.get(field)!r} != {treatment_record.get(field)!r}"
                )

    baseline_problem = _problem_metrics(
        baseline,
        samples_per_problem=samples_per_problem,
    )
    treatment_problem = _problem_metrics(
        treatment,
        samples_per_problem=samples_per_problem,
    )
    if baseline_problem.keys() != treatment_problem.keys():
        raise ValueError("baseline and treatment problem IDs do not match")

    changes = Counter()
    transition_rows: dict[
        str,
        list[tuple[dict[str, Any], dict[str, Any]]],
    ] = defaultdict(list)
    for record_key, base_record in baseline_by_key.items():
        base_correct = bool(base_record["correct"])
        treatment_record = treatment_by_key[record_key]
        treatment_correct = bool(treatment_record["correct"])
        if base_correct and treatment_correct:
            transition = "both_correct"
        elif not base_correct and not treatment_correct:
            transition = "both_wrong"
        elif base_correct:
            transition = "degraded"
        else:
            transition = "improved"
        changes[transition] += 1
        transition_rows[transition].append((base_record, treatment_record))

    generation_diagnostics, transition_diagnostics = _generation_diagnostics(
        baseline_by_key,
        treatment_by_key,
        transition_rows,
    )

    suffix = str(samples_per_problem)
    intervals = _paired_bootstrap(
        baseline_problem,
        treatment_problem,
        bootstrap_samples=bootstrap_samples,
    )
    first_baseline = baseline[0]
    first_treatment = treatment[0]
    return {
        "schema_version": 1,
        "comparison_protocol": "paired-problem-cluster-bootstrap-v1",
        "evaluation_protocol": (
            "official" if samples_per_problem == 12 else "development"
        ),
        "sample_subset_protocol": (
            "first-n-paired-samples-v1" if take_first_samples else "exact-run-v1"
        ),
        "pairing_verified": True,
        "model": first_baseline["model"],
        "benchmark": first_baseline["benchmark"],
        "baseline_method": first_baseline["method"],
        "baseline_checkpoint": first_baseline["checkpoint"],
        "treatment_method": first_treatment["method"],
        "treatment_checkpoint": first_treatment["checkpoint"],
        "num_problems": len(baseline_problem),
        "samples_per_problem": samples_per_problem,
        f"avg_at_{suffix}": _metric_triplet(
            baseline_problem, treatment_problem, "avg"
        ),
        f"pass_at_{suffix}": _metric_triplet(
            baseline_problem, treatment_problem, "pass"
        ),
        f"maj_at_{suffix}": _metric_triplet(
            baseline_problem, treatment_problem, "maj"
        ),
        "delta_bootstrap_95ci": {
            f"avg_at_{suffix}": intervals["avg"],
            f"pass_at_{suffix}": intervals["pass"],
            f"maj_at_{suffix}": intervals["maj"],
        },
        "paired_sample_changes": {
            key: changes[key]
            for key in ("both_correct", "both_wrong", "degraded", "improved")
        },
        "generation_diagnostics": generation_diagnostics,
        "transition_diagnostics": transition_diagnostics,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two paired, official-scored math runs."
    )
    parser.add_argument("--baseline-input", required=True, nargs="+", type=Path)
    parser.add_argument("--baseline-grades", required=True, type=Path)
    parser.add_argument("--treatment-input", required=True, nargs="+", type=Path)
    parser.add_argument("--treatment-grades", required=True, type=Path)
    parser.add_argument("--samples-per-problem", required=True, type=int)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--take-first-samples", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    comparison = compare_paired_math(
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
        samples_per_problem=args.samples_per_problem,
        bootstrap_samples=args.bootstrap_samples,
        take_first_samples=args.take_first_samples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
