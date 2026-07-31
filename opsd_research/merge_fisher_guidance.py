"""Merge disjoint answer-free guidance cache shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .fisher_guidance import CACHE_SCHEMA_VERSION, load_guidance_ensemble


_IDENTITY_FIELDS = (
    "cache_kind",
    "answer_access",
    "reference_solution_access",
    "guidance_input_protocol",
    "plans_per_problem",
    "builder_model",
    "builder_model_revision",
    "training_dataset_revision",
    "data_sources",
    "selection_seed",
    "builder_seed",
    "builder_attempts",
    "requested_global_records",
    "num_shards",
)


def merge_guidance_shards(
    manifests: list[Path],
    *,
    output: Path,
    manifest_output: Path,
) -> dict[str, object]:
    if not manifests:
        raise ValueError("at least one guidance shard is required")
    if output.exists() or manifest_output.exists():
        raise ValueError("refusing to overwrite merged guidance")
    loaded = []
    for path in manifests:
        ensembles, metadata = load_guidance_ensemble(path)
        loaded.append((path, ensembles, metadata))
    loaded.sort(key=lambda item: int(item[2]["shard_id"]))
    reference = loaded[0][2]
    for path, _ensembles, metadata in loaded[1:]:
        for field in _IDENTITY_FIELDS:
            if metadata.get(field) != reference.get(field):
                raise ValueError(f"guidance shard {path} mismatches {field}")
    expected_shards = int(reference["num_shards"])
    shard_ids = [int(metadata["shard_id"]) for _, _, metadata in loaded]
    if shard_ids != list(range(expected_shards)):
        raise ValueError(
            f"guidance shards must be exactly 0..{expected_shards - 1}"
        )

    records: dict[int, dict[str, object]] = {}
    rejected = 0
    requested = 0
    accepted_by_domain: dict[str, int] = {}
    for path, ensembles, metadata in loaded:
        rejected += int(metadata.get("rejected_records", 0))
        requested += int(metadata["requested_records"])
        for domain, count in metadata.get("accepted_by_domain", {}).items():
            accepted_by_domain[str(domain)] = (
                accepted_by_domain.get(str(domain), 0) + int(count)
            )
        records_path = path.parent / str(metadata["records_file"])
        by_index = {
            int(record["source_index"]): record
            for record in (
                json.loads(line)
                for line in records_path.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            )
        }
        if set(by_index) != set(ensembles):
            raise ValueError(f"guidance shard {path} record mismatch")
        overlap = set(records).intersection(by_index)
        if overlap:
            raise ValueError(
                f"duplicate guidance source index {min(overlap)}"
            )
        records.update(by_index)
    if requested != int(reference["requested_global_records"]):
        raise ValueError("merged guidance requested-record count mismatch")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for index in sorted(records):
            handle.write(
                json.dumps(
                    records[index], ensure_ascii=False, sort_keys=True
                )
                + "\n"
            )
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    result: dict[str, object] = {
        "schema_version": CACHE_SCHEMA_VERSION,
        **{field: reference.get(field) for field in _IDENTITY_FIELDS},
        "accepted_records": len(records),
        "accepted_by_domain": dict(sorted(accepted_by_domain.items())),
        "rejected_records": rejected,
        "requested_records": requested,
        "records_file": output.name,
        "records_sha256": digest,
        "source_shards": [str(path) for path, _, _ in loaded],
    }
    manifest_output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    # Re-read through the exact training validator before publishing.
    load_guidance_ensemble(manifest_output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest-output", required=True, type=Path)
    args = parser.parse_args()
    result = merge_guidance_shards(
        args.manifest,
        output=args.output,
        manifest_output=args.manifest_output,
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
