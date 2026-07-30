"""Merge independently built representative GRAF cache shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .graf_cache import validate_graph_cache_manifest
from .records import read_jsonl


_IDENTITY_FIELDS = (
    "builder_model",
    "builder_model_revision",
    "training_dataset_revision",
    "builder_seed",
    "builder_attempts",
    "max_forks",
    "max_actions_per_fork",
    "graph_budget",
    "teacher_critique",
    "guidance_input_protocol",
    "answer_leakage_protocol",
    "selection_protocol",
    "selection_seed",
    "requested_global_examples",
    "num_shards",
)


def merge_graph_cache_shards(
    manifests: list[Path],
    *,
    output: Path,
    manifest_output: Path,
) -> dict[str, Any]:
    """Validate, join, and checksum a complete set of disjoint cache shards."""
    if not manifests:
        raise ValueError("at least one cache shard is required")
    if output.exists() or manifest_output.exists():
        raise ValueError("refusing to overwrite a merged graph cache")
    loaded = [
        (path, validate_graph_cache_manifest(path))
        for path in manifests
    ]
    loaded.sort(key=lambda item: int(item[1].get("shard_id", -1)))
    reference = loaded[0][1]
    for path, manifest in loaded[1:]:
        for field in _IDENTITY_FIELDS:
            if manifest.get(field) != reference.get(field):
                raise ValueError(
                    f"cache shard {path} has mismatched {field}"
                )
    expected_shards = int(reference["num_shards"])
    shard_ids = [int(manifest["shard_id"]) for _path, manifest in loaded]
    if shard_ids != list(range(expected_shards)):
        raise ValueError(
            f"cache shards must be exactly 0..{expected_shards - 1}"
        )

    records: dict[int, dict[str, Any]] = {}
    for path, manifest in loaded:
        cache_path = Path(str(manifest["cache"]))
        if not cache_path.is_absolute():
            cache_path = path.parent / cache_path
        shard_records = read_jsonl(cache_path)
        if len(shard_records) != int(manifest["requested_examples"]):
            raise ValueError(f"cache shard {path} has a record-count mismatch")
        for record in shard_records:
            index = int(record["example_index"])
            if index in records:
                raise ValueError(f"duplicate source example_index {index}")
            records[index] = record

    requested = len(records)
    expected_requested = int(reference["requested_global_examples"])
    if requested != expected_requested:
        raise ValueError(
            f"merged cache has {requested} records, expected {expected_requested}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for index in sorted(records):
            handle.write(
                json.dumps(records[index], ensure_ascii=False, sort_keys=True)
                + "\n"
            )
    accepted = sum(bool(record.get("accepted")) for record in records.values())
    result = {
        "schema_version": 1,
        "cache": str(output),
        "cache_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "requested_examples": requested,
        "accepted_examples": accepted,
        "rejected_examples": requested - accepted,
        **{field: reference.get(field) for field in _IDENTITY_FIELDS},
        "teacher_critique_applied": sum(
            int(manifest.get("teacher_critique_applied", 0))
            for _path, manifest in loaded
        ),
        "teacher_critique_rejected": sum(
            int(manifest.get("teacher_critique_rejected", 0))
            for _path, manifest in loaded
        ),
        "source_shards": [str(path) for path, _manifest in loaded],
    }
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest-output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = merge_graph_cache_shards(
        args.manifest,
        output=args.output,
        manifest_output=args.manifest_output,
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
