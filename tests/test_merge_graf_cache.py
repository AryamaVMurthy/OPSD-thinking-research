from __future__ import annotations

import hashlib
import json
from pathlib import Path

from opsd_research.merge_graf_cache import merge_graph_cache_shards


def _shard(root: Path, shard_id: int, records: list[dict]) -> Path:
    cache = root / f"graphs-{shard_id}.jsonl"
    cache.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    accepted = sum(bool(record["accepted"]) for record in records)
    manifest = root / f"manifest-{shard_id}.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cache": str(cache),
                "cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
                "requested_examples": len(records),
                "requested_global_examples": 4,
                "accepted_examples": accepted,
                "rejected_examples": len(records) - accepted,
                "builder_model": "Qwen/Qwen3-4B",
                "builder_model_revision": "revision",
                "training_dataset_revision": "dataset-revision",
                "builder_seed": 42,
                "builder_attempts": 3,
                "max_forks": 6,
                "max_actions_per_fork": 6,
                "graph_budget": 24,
                "teacher_critique": False,
                "teacher_critique_applied": 0,
                "teacher_critique_rejected": 0,
                "selection_protocol": "content-hash-uniform-v1",
                "selection_seed": 73,
                "shard_id": shard_id,
                "num_shards": 2,
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_merges_disjoint_cache_shards_in_source_index_order(tmp_path: Path):
    first = _shard(
        tmp_path,
        0,
        [
            {"example_index": 9, "accepted": True},
            {"example_index": 3, "accepted": False},
        ],
    )
    second = _shard(
        tmp_path,
        1,
        [
            {"example_index": 7, "accepted": True},
            {"example_index": 1, "accepted": True},
        ],
    )
    output = tmp_path / "merged.jsonl"
    manifest = tmp_path / "merged-manifest.json"

    result = merge_graph_cache_shards(
        [first, second],
        output=output,
        manifest_output=manifest,
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["example_index"] for row in rows] == [1, 3, 7, 9]
    assert result["requested_examples"] == 4
    assert result["accepted_examples"] == 3
    assert result["rejected_examples"] == 1
    assert result["source_shards"] == [str(first), str(second)]
