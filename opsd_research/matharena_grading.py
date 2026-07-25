from __future__ import annotations

import hashlib
import importlib
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from .generation_common import extract_last_boxed


def _matharena_source() -> Path:
    return Path(__file__).resolve().parents[1] / "third_party" / "matharena" / "src"


def _official_parser() -> tuple[Any, Any, Any]:
    source = _matharena_source()
    if not (source / "matharena" / "parser.py").is_file():
        raise RuntimeError(
            "official MathArena checkout is missing; initialize third_party/matharena"
        )
    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    parser = importlib.import_module("matharena.parser")
    parser_path = Path(parser.__file__).resolve()
    if not parser_path.is_relative_to(source.resolve()):
        raise RuntimeError(
            f"loaded MathArena parser from {parser_path}, expected pinned source {source}"
        )

    return parser.extract_answer, parser.parse_answer, parser.check_answers


def grade_response(
    response: str,
    ground_truth: str,
    *,
    strict_parsing: bool,
) -> dict[str, Any]:
    """Grade one full response with the pinned official MathArena parser."""
    extract_answer, parse_answer, check_answers = _official_parser()
    list_answer = "," in str(ground_truth)
    predicted, warning = extract_answer(
        response,
        strict_parsing=strict_parsing,
        parse=True,
        list_answer=list_answer,
        typed_delimiters=True,
    )
    gold, _ = parse_answer(
        str(ground_truth),
        list_answer=list_answer,
        typed_delimiters=True,
    )
    correct = bool(check_answers(predicted, gold))
    try:
        serialized = None if predicted is None else str(predicted)
    except (OverflowError, ValueError):
        raw_answer = extract_last_boxed(response)
        hash_source = response if raw_answer is None else raw_answer
        digest = hashlib.sha256(hash_source.encode("utf-8")).hexdigest()
        serialized = f"<unrenderable:{type(predicted).__name__}:{digest}>"
    return {
        "predicted_answer": serialized,
        "correct": correct,
        "warning": warning.value,
    }


def rescore_record(
    record: dict[str, Any],
    *,
    strict_parsing: bool,
) -> dict[str, Any]:
    """Return an auditable grading sidecar without changing the raw record."""
    grade = grade_response(
        str(record["response"]),
        str(record["ground_truth"]),
        strict_parsing=strict_parsing,
    )
    identity_fields = (
        "model",
        "method",
        "checkpoint",
        "benchmark",
        "problem_id",
        "sample_index",
        "seed",
        "prompt_hash",
    )
    sidecar = {
        "schema_version": 1,
        **{field: record.get(field) for field in identity_fields},
        "response_sha256": hashlib.sha256(
            str(record["response"]).encode("utf-8")
        ).hexdigest(),
        "legacy_predicted_answer": record.get("predicted_answer"),
        "legacy_correct": bool(record.get("correct", False)),
        "official_predicted_answer": grade["predicted_answer"],
        "official_correct": grade["correct"],
        "official_warning": grade["warning"],
    }
    return sidecar


def _record_key(record: dict[str, Any]) -> tuple[str, int]:
    return str(record["problem_id"]), int(record["sample_index"])


def _bootstrap_problem_mean(
    values: dict[str, list[bool]],
    *,
    seed: int = 42,
) -> list[float]:
    rng = random.Random(seed)
    problem_ids = sorted(values)
    estimates = []
    for _ in range(5000):
        drawn = [rng.choice(problem_ids) for _ in problem_ids]
        estimates.append(
            mean(mean(int(value) for value in values[problem_id]) for problem_id in drawn)
        )
    return sorted(estimates)


def summarize_rescored(
    records: list[dict[str, Any]],
    sidecars: list[dict[str, Any]],
    *,
    samples_per_problem: int,
    grader_revision: str,
    strict_parsing: bool,
) -> dict[str, Any]:
    """Validate raw/sidecar pairing and summarize official MathArena grades."""
    if not records:
        raise ValueError("cannot summarize an empty record set")
    raw_by_key = {_record_key(record): record for record in records}
    grade_by_key = {_record_key(sidecar): sidecar for sidecar in sidecars}
    if len(raw_by_key) != len(records):
        raise ValueError("duplicate raw generation keys")
    if len(grade_by_key) != len(sidecars):
        raise ValueError("duplicate official grading sidecar keys")
    if raw_by_key.keys() != grade_by_key.keys():
        raise ValueError("raw generation and official grading keys do not match")

    for record_key, record in raw_by_key.items():
        sidecar = grade_by_key[record_key]
        response_hash = hashlib.sha256(
            str(record["response"]).encode("utf-8")
        ).hexdigest()
        if sidecar["response_sha256"] != response_hash:
            raise ValueError(f"response hash mismatch for key {record_key}")

    by_problem: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for record_key in sorted(raw_by_key):
        by_problem[record_key[0]].append(
            (raw_by_key[record_key], grade_by_key[record_key])
        )
    for problem_id, problem_records in by_problem.items():
        if len(problem_records) != samples_per_problem:
            raise ValueError(
                f"problem {problem_id}: expected {samples_per_problem} samples, "
                f"found {len(problem_records)}"
            )

    correct_by_problem = {
        problem_id: [bool(sidecar["official_correct"]) for _, sidecar in rows]
        for problem_id, rows in by_problem.items()
    }
    majority = []
    for rows in by_problem.values():
        formatted = [
            str(sidecar["official_predicted_answer"])
            for _, sidecar in rows
            if sidecar["official_predicted_answer"] is not None
        ]
        if not formatted:
            majority.append(False)
            continue
        counts = Counter(formatted)
        highest = max(counts.values())
        # Samples were sorted above, so ties resolve to the earliest sample.
        answer = next(value for value in formatted if counts[value] == highest)
        majority.append(
            next(
                bool(sidecar["official_correct"])
                for _, sidecar in rows
                if str(sidecar["official_predicted_answer"]) == answer
            )
        )

    all_pairs = [pair for rows in by_problem.values() for pair in rows]
    official_values = [bool(sidecar["official_correct"]) for _, sidecar in all_pairs]
    bootstrap = _bootstrap_problem_mean(correct_by_problem)
    lengths = sorted(int(record["output_tokens"]) for record, _ in all_pairs)
    changed = [
        (bool(sidecar["legacy_correct"]), bool(sidecar["official_correct"]))
        for _, sidecar in all_pairs
        if bool(sidecar["legacy_correct"]) != bool(sidecar["official_correct"])
    ]
    first = records[0]
    metric_suffix = str(samples_per_problem)
    warning_counts = Counter(
        str(sidecar["official_warning"]) for _, sidecar in all_pairs
    )
    return {
        "schema_version": 1,
        "scoring_protocol": "official-matharena-parser-v1",
        "grader_revision": grader_revision,
        "strict_parsing": strict_parsing,
        "typed_delimited_answers": True,
        "model": first["model"],
        "model_revision": first["model_revision"],
        "method": first["method"],
        "checkpoint": first["checkpoint"],
        "adapter_sha256": first.get("adapter_sha256"),
        "seed_protocol": first.get("seed_protocol"),
        "benchmark": first["benchmark"],
        "dataset_revision": first["dataset_revision"],
        "num_problems": len(by_problem),
        "samples_per_problem": samples_per_problem,
        f"avg_at_{metric_suffix}": mean(int(value) for value in official_values),
        f"pass_at_{metric_suffix}": mean(
            any(values) for values in correct_by_problem.values()
        ),
        f"maj_at_{metric_suffix}": mean(majority),
        "bootstrap_95ci": [bootstrap[125], bootstrap[4874]],
        "official_format_rate": mean(
            sidecar["official_predicted_answer"] is not None
            for _, sidecar in all_pairs
        ),
        "boxed_format_rate": mean(
            bool(record["formatted"]) for record, _ in all_pairs
        ),
        "nonempty_thinking_rate": mean(
            bool(record["nonempty_thinking"]) for record, _ in all_pairs
        ),
        "length_cutoff_rate": mean(
            record["finish_reason"] == "length" for record, _ in all_pairs
        ),
        "official_warning_counts": dict(sorted(warning_counts.items())),
        "grade_changes": {
            "changed": len(changed),
            "legacy_false_negative": sum(
                not legacy and official for legacy, official in changed
            ),
            "legacy_false_positive": sum(
                legacy and not official for legacy, official in changed
            ),
        },
        "output_tokens": {
            "mean": mean(lengths),
            "median": median(lengths),
            "p90": lengths[int(0.9 * (len(lengths) - 1))],
            "max": max(lengths),
        },
    }
