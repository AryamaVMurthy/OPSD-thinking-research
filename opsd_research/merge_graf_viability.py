"""Merge independently generated viability shards into one immutable cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .graf_cache import validate_graph_cache_manifest
from .records import append_jsonl, read_jsonl


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-manifest", required=True, type=Path)
    parser.add_argument("--shard", required=True, type=Path, action="append")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--samples-per-action", required=True, type=int)
    parser.add_argument("--temperature", required=True, type=float)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--seed", required=True, type=int)
    return parser.parse_args()


def main() -> None:
    args = _args()
    if args.output.exists() or args.manifest.exists():
        raise SystemExit("refusing to overwrite an immutable viability cache or manifest")
    graph_manifest = validate_graph_cache_manifest(args.graph_manifest)
    graph_path = Path(str(graph_manifest["cache"]))
    if not graph_path.is_absolute():
        graph_path = args.graph_manifest.parent / graph_path
    expected = {
        int(record["example_index"]): str(record["graph_sha256"])
        for record in read_jsonl(graph_path)
        if record.get("accepted")
    }
    records: dict[int, dict] = {}
    for shard in args.shard:
        if not shard.is_file():
            raise SystemExit(f"viability shard does not exist: {shard}")
        for record in read_jsonl(shard):
            index = int(record.get("example_index", -1))
            if index in records:
                raise SystemExit(f"duplicate viability target for example {index}")
            if expected.get(index) != record.get("graph_sha256"):
                raise SystemExit(f"viability target {index} is not joined to the graph cache")
            if int(record.get("samples_per_action", -1)) != args.samples_per_action:
                raise SystemExit("viability shard samples-per-action mismatch")
            if float(record.get("temperature", -1)) != args.temperature:
                raise SystemExit("viability shard temperature mismatch")
            if record.get("model") != args.model or record.get("model_revision") != args.model_revision:
                raise SystemExit("viability shard model pin mismatch")
            records[index] = record
    if not records:
        raise SystemExit("no viability records to merge")
    for index in sorted(records):
        append_jsonl(args.output, records[index])
    manifest = {
        "schema_version": 1,
        "graph_cache_sha256": graph_manifest["cache_sha256"],
        "viability_cache": str(args.output),
        "viability_cache_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "examples": len(records),
        "samples_per_action": args.samples_per_action,
        "temperature": args.temperature,
        "model": args.model,
        "model_revision": args.model_revision,
        "seed": args.seed,
        "shards": [str(path) for path in args.shard],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
