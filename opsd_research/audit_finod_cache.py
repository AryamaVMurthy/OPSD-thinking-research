"""Replay a FiNOD graph cache through the exact training-time safety path."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .answer_masking import (
    ANSWER_LEAKAGE_PROTOCOL,
    PROBLEM_ONLY_GUIDANCE_PROTOCOL,
)
from .finod_dataset import finod_training_row, remove_graph_identifiers
from .graf_cache import validate_graph_cache_manifest
from .graf_scaffold_dataset import accepted_scaffolds
from .records import read_jsonl
from .training_data import DATASET_REVISION


def audit_finod_cache(
    manifest_path: str | Path,
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Fail closed unless every accepted row is safe for the training adapter."""
    manifest = validate_graph_cache_manifest(manifest_path)
    required = {
        "training_dataset_revision": DATASET_REVISION,
        "guidance_input_protocol": PROBLEM_ONLY_GUIDANCE_PROTOCOL,
        "answer_leakage_protocol": ANSWER_LEAKAGE_PROTOCOL,
    }
    for field, expected in required.items():
        if manifest.get(field) != expected:
            raise ValueError(
                f"FiNOD cache requires {field}={expected!r}, "
                f"observed {manifest.get(field)!r}"
            )

    cache_path = Path(str(manifest["cache"]))
    if not cache_path.is_absolute():
        cache_path = Path(manifest_path).parent / cache_path
    records = read_jsonl(cache_path)
    if len(records) != int(manifest["requested_examples"]):
        raise ValueError("FiNOD cache record count does not match its manifest")
    scaffolds = accepted_scaffolds(manifest_path)
    accepted_sources: Counter[str] = Counter()
    rejected_reasons: Counter[str] = Counter()
    accepted_seen: set[int] = set()
    for record in records:
        for field, expected in required.items():
            if field == "training_dataset_revision":
                continue
            if record.get(field) != expected:
                raise ValueError(
                    f"cache record has mismatched {field}: "
                    f"{record.get(field)!r}"
                )
        if not record.get("accepted"):
            rejected_reasons[str(record.get("reject_reason", "unknown"))] += 1
            continue
        index = int(record["example_index"])
        if index in accepted_seen:
            raise ValueError(f"duplicate accepted FiNOD source index {index}")
        accepted_seen.add(index)
        if not 0 <= index < len(rows):
            raise ValueError(f"FiNOD source index {index} is outside the dataset")
        row = rows[index]
        question = str(row["question"])
        expected_problem_sha = hashlib.sha256(
            question.encode("utf-8")
        ).hexdigest()
        if record.get("problem_sha256") != expected_problem_sha:
            raise ValueError(
                f"FiNOD problem hash mismatch at source index {index}"
            )
        graph_payload = record.get("graph")
        if not isinstance(graph_payload, dict):
            raise ValueError(f"FiNOD graph is missing at source index {index}")
        graph_sha = hashlib.sha256(
            json.dumps(
                graph_payload, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        if record.get("graph_sha256") != graph_sha:
            raise ValueError(
                f"FiNOD graph hash mismatch at source index {index}"
            )
        try:
            finod_training_row(
                question=question,
                reference_solution=str(row["response"]),
                answer_masked_guide=remove_graph_identifiers(
                    scaffolds[index]
                ),
                source_index=index,
            )
        except ValueError as error:
            raise ValueError(
                f"FiNOD training safety check failed at source index "
                f"{index}: {error}"
            ) from error
        accepted_sources[str(row.get("data_source") or "unknown")] += 1
    if accepted_seen != set(scaffolds):
        raise ValueError("accepted cache records and rendered scaffolds disagree")
    return {
        "schema_version": 1,
        "audit_protocol": "finod-training-path-replay-v1",
        "cache_sha256": manifest["cache_sha256"],
        "training_dataset_revision": DATASET_REVISION,
        "guidance_input_protocol": PROBLEM_ONLY_GUIDANCE_PROTOCOL,
        "answer_leakage_protocol": ANSWER_LEAKAGE_PROTOCOL,
        "accepted_examples_replayed": len(accepted_seen),
        "rejected_examples": len(records) - len(accepted_seen),
        "accepted_by_data_source": dict(sorted(accepted_sources.items())),
        "rejected_by_reason": dict(sorted(rejected_reasons.items())),
        "training_safe": True,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit a problem-only FiNOD graph cache"
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite FiNOD audit {args.output}")
    from .training_data import load_math_cot_20k

    rows = load_math_cot_20k(heldout_fraction=0.0)["train"]
    summary = audit_finod_cache(args.manifest, rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
