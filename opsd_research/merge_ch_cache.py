"""Merge complete CH-OPSD GPU shards into one immutable verified cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def merge_ch_cache_shards(
    *,
    shards: list[Path],
    output: Path,
    manifest: Path,
    requested_examples: int,
    blind_attempts_per_problem: int,
    builder_model: str,
    builder_model_revision: str,
    training_dataset_revision: str,
    builder_seed: int,
) -> dict[str, Any]:
    """Validate shard coverage and write one content-addressed cache."""
    if not shards:
        raise ValueError("at least one CH cache shard is required")
    if requested_examples < 1:
        raise ValueError("requested_examples must be positive")
    if blind_attempts_per_problem not in {2, 3}:
        raise ValueError("CH-OPSD requires two or three blind attempts")
    if output.exists() or manifest.exists():
        raise ValueError("refusing to alter an existing CH cache or manifest")

    num_shards = len(shards)
    records: dict[int, dict[str, Any]] = {}
    for expected_shard_id, shard in enumerate(shards):
        for line_number, line in enumerate(
            shard.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON in {shard} at line {line_number}"
                ) from error
            if not isinstance(record, dict) or record.get("schema_version") != 1:
                raise ValueError(f"invalid CH record in {shard} at line {line_number}")
            index = int(record.get("example_index", -1))
            if (
                int(record.get("num_shards", -1)) != num_shards
                or int(record.get("shard_id", -1)) != expected_shard_id
                or index % num_shards != expected_shard_id
            ):
                raise ValueError("CH cache shard identity or routing is inconsistent")
            if index in records:
                raise ValueError(f"duplicate CH cache example_index {index}")
            if int(record.get("blind_attempt_count", -1)) != blind_attempts_per_problem:
                raise ValueError("CH record blind-attempt count is inconsistent")
            if bool(record.get("accepted")):
                if not str(record.get("teacher_dossier", "")).strip():
                    raise ValueError("accepted CH record has no teacher dossier")
            elif not str(record.get("rejection_reason", "")).strip():
                raise ValueError("rejected CH record has no rejection reason")
            records[index] = record

    expected_indices = set(range(requested_examples))
    if set(records) != expected_indices:
        missing = sorted(expected_indices.difference(records))
        unexpected = sorted(set(records).difference(expected_indices))
        raise ValueError(
            f"CH shard coverage mismatch; missing={missing[:5]}, "
            f"unexpected={unexpected[:5]}"
        )
    accepted = sum(bool(record["accepted"]) for record in records.values())
    rejected = requested_examples - accepted
    if accepted == 0:
        raise ValueError("CH cache contained no accepted teacher dossiers")

    output.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(
        json.dumps(records[index], ensure_ascii=False, sort_keys=True) + "\n"
        for index in range(requested_examples)
    ).encode("utf-8")
    output.write_bytes(content)
    payload = {
        "schema_version": 1,
        "cache": str(output.resolve()),
        "cache_sha256": hashlib.sha256(content).hexdigest(),
        "requested_examples": requested_examples,
        "accepted_examples": accepted,
        "rejected_examples": rejected,
        "blind_attempts_per_problem": blind_attempts_per_problem,
        "audit_format": "natural_language_v1",
        "schema_based_selection": False,
        "student_answer_context": False,
        "teacher_reference_context": True,
        "builder_model": builder_model,
        "builder_model_revision": builder_model_revision,
        "training_dataset_revision": training_dataset_revision,
        "builder_seed": builder_seed,
        "num_shards": num_shards,
        "shards": [str(path.resolve()) for path in shards],
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--requested-examples", required=True, type=int)
    parser.add_argument("--blind-attempts-per-problem", required=True, type=int)
    parser.add_argument("--builder-model", required=True)
    parser.add_argument("--builder-model-revision", required=True)
    parser.add_argument("--training-dataset-revision", required=True)
    parser.add_argument("--builder-seed", required=True, type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        payload = merge_ch_cache_shards(
            shards=args.shards,
            output=args.output,
            manifest=args.manifest,
            requested_examples=args.requested_examples,
            blind_attempts_per_problem=args.blind_attempts_per_problem,
            builder_model=args.builder_model,
            builder_model_revision=args.builder_model_revision,
            training_dataset_revision=args.training_dataset_revision,
            builder_seed=args.builder_seed,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
