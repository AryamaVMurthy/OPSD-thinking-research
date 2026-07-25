import hashlib
import json
import sys
from pathlib import Path

from opsd_research import merge_graf_viability


def test_merges_verified_nonoverlapping_shards(tmp_path: Path, monkeypatch) -> None:
    graphs = tmp_path / "graphs.jsonl"
    graph_rows = [
        {"accepted": True, "example_index": 1, "graph_sha256": "a"},
        {"accepted": True, "example_index": 2, "graph_sha256": "b"},
    ]
    graphs.write_text("".join(json.dumps(row) + "\n" for row in graph_rows), encoding="utf-8")
    graph_manifest = tmp_path / "graphs-manifest.json"
    graph_manifest.write_text(json.dumps({
        "schema_version": 1, "cache": "graphs.jsonl",
        "cache_sha256": hashlib.sha256(graphs.read_bytes()).hexdigest(),
        "requested_examples": 2, "accepted_examples": 2, "rejected_examples": 0,
    }), encoding="utf-8")
    shards = []
    for index, digest in ((1, "a"), (2, "b")):
        shard = tmp_path / f"part-{index}.jsonl"
        shard.write_text(json.dumps({
            "example_index": index, "graph_sha256": digest, "samples_per_action": 1,
            "temperature": 1.0, "model": "Qwen/Qwen3-4B", "model_revision": "pin",
            "fork_targets": [{"fork_id": "f", "action_ids": ["x"], "target": [1.0]}],
        }) + "\n", encoding="utf-8")
        shards.append(shard)
    output, manifest = tmp_path / "merged.jsonl", tmp_path / "merged-manifest.json"
    monkeypatch.setattr(sys, "argv", [
        "merge_graf_viability", "--graph-manifest", str(graph_manifest),
        "--shard", str(shards[0]), "--shard", str(shards[1]), "--output", str(output),
        "--manifest", str(manifest), "--samples-per-action", "1", "--temperature", "1.0",
        "--model", "Qwen/Qwen3-4B", "--model-revision", "pin", "--seed", "42",
    ])
    merge_graf_viability.main()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["examples"] == 2
    assert [json.loads(line)["example_index"] for line in output.read_text().splitlines()] == [1, 2]
