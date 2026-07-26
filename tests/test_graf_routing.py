import hashlib
import json
from pathlib import Path

from opsd_research.graf_routing import load_routing_targets
from opsd_research.graf_actions import ASSISTANT_ACTION_PREFIX_PROTOCOL


def test_joins_matching_immutable_graph_and_viability_caches(tmp_path: Path) -> None:
    graph_cache = tmp_path / "graphs.jsonl"
    graph = {
        "accepted": True, "example_index": 3, "graph_sha256": "graph-hash",
        "graph": {"forks": [{"actions": [
            {"action_id": "a", "description": "factor", "status": "viable"},
            {"action_id": "b", "description": "substitute", "status": "risky"},
        ]}]},
    }
    graph_cache.write_text(json.dumps(graph) + "\n", encoding="utf-8")
    graph_manifest = tmp_path / "graph-manifest.json"
    graph_digest = hashlib.sha256(graph_cache.read_bytes()).hexdigest()
    graph_manifest.write_text(json.dumps({
        "schema_version": 1, "cache": "graphs.jsonl", "cache_sha256": graph_digest,
        "requested_examples": 1, "accepted_examples": 1, "rejected_examples": 0,
    }), encoding="utf-8")
    viability_cache = tmp_path / "viability.jsonl"
    viability_cache.write_text(json.dumps({
        "example_index": 3, "graph_sha256": "graph-hash",
        "forced_prefix_protocol": ASSISTANT_ACTION_PREFIX_PROTOCOL,
        "fork_targets": [{"fork_id": "f", "action_ids": ["a", "b"], "target": [0.75, 0.25]}],
    }) + "\n", encoding="utf-8")
    viability_manifest = tmp_path / "viability-manifest.json"
    viability_manifest.write_text(json.dumps({
        "schema_version": 1, "graph_cache_sha256": graph_digest,
        "viability_cache": "viability.jsonl",
        "viability_cache_sha256": hashlib.sha256(viability_cache.read_bytes()).hexdigest(),
        "examples": 1, "forced_prefix_protocol": ASSISTANT_ACTION_PREFIX_PROTOCOL,
    }), encoding="utf-8")

    targets = load_routing_targets(graph_manifest, viability_manifest)
    assert targets[3][0].actions[0].description == "factor"
    assert targets[3][0].actions[1].target_probability == 0.25


def test_retains_example_but_filters_uniform_forks(tmp_path: Path) -> None:
    graph_cache = tmp_path / "graphs.jsonl"
    graph = {
        "accepted": True, "example_index": 3, "graph_sha256": "graph-hash",
        "graph": {"forks": [{"actions": [
            {"action_id": "a", "description": "factor", "status": "viable"},
            {"action_id": "b", "description": "substitute", "status": "risky"},
        ]}]},
    }
    graph_cache.write_text(json.dumps(graph) + "\n", encoding="utf-8")
    digest = hashlib.sha256(graph_cache.read_bytes()).hexdigest()
    graph_manifest = tmp_path / "graph-manifest.json"
    graph_manifest.write_text(json.dumps({"schema_version": 1, "cache": "graphs.jsonl", "cache_sha256": digest, "requested_examples": 1, "accepted_examples": 1, "rejected_examples": 0}), encoding="utf-8")
    viability_cache = tmp_path / "viability.jsonl"
    viability_cache.write_text(json.dumps({"example_index": 3, "graph_sha256": "graph-hash", "forced_prefix_protocol": ASSISTANT_ACTION_PREFIX_PROTOCOL, "fork_targets": [{"fork_id": "f", "action_ids": ["a", "b"], "target": [0.5, 0.5]}]}) + "\n", encoding="utf-8")
    viability_manifest = tmp_path / "viability-manifest.json"
    viability_manifest.write_text(json.dumps({"schema_version": 1, "graph_cache_sha256": digest, "viability_cache": "viability.jsonl", "viability_cache_sha256": hashlib.sha256(viability_cache.read_bytes()).hexdigest(), "examples": 1, "forced_prefix_protocol": ASSISTANT_ACTION_PREFIX_PROTOCOL}), encoding="utf-8")

    targets = load_routing_targets(graph_manifest, viability_manifest, min_target_margin=0.15)
    assert targets == {3: ()}
