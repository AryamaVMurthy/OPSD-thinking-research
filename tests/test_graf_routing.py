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


def test_information_filter_retains_two_viable_actions_and_suppresses_uniform(tmp_path: Path) -> None:
    graph_cache = tmp_path / "graphs.jsonl"
    graph = {
        "accepted": True, "example_index": 3, "graph_sha256": "graph-hash",
        "graph": {"forks": [{"actions": [
            {"action_id": "a", "description": "factor", "status": "viable"},
            {"action_id": "b", "description": "substitute", "status": "viable"},
            {"action_id": "c", "description": "guess", "status": "risky"},
        ]}]},
    }
    graph_cache.write_text(json.dumps(graph) + "\n", encoding="utf-8")
    digest = hashlib.sha256(graph_cache.read_bytes()).hexdigest()
    graph_manifest = tmp_path / "graph-manifest.json"
    graph_manifest.write_text(json.dumps({"schema_version": 1, "cache": "graphs.jsonl", "cache_sha256": digest, "requested_examples": 1, "accepted_examples": 1, "rejected_examples": 0}), encoding="utf-8")
    viability_cache = tmp_path / "viability.jsonl"
    viability_cache.write_text(json.dumps({"example_index": 3, "graph_sha256": "graph-hash", "forced_prefix_protocol": ASSISTANT_ACTION_PREFIX_PROTOCOL, "fork_targets": [
        {"fork_id": "informative", "action_ids": ["a", "b", "c"], "target": [0.42232, 0.42232, 0.15536]},
        {"fork_id": "uniform", "action_ids": ["a", "b", "c"], "target": [1 / 3, 1 / 3, 1 / 3]},
    ]}) + "\n", encoding="utf-8")
    viability_manifest = tmp_path / "viability-manifest.json"
    viability_manifest.write_text(json.dumps({"schema_version": 1, "graph_cache_sha256": digest, "viability_cache": "viability.jsonl", "viability_cache_sha256": hashlib.sha256(viability_cache.read_bytes()).hexdigest(), "examples": 1, "forced_prefix_protocol": ASSISTANT_ACTION_PREFIX_PROTOCOL}), encoding="utf-8")

    targets = load_routing_targets(graph_manifest, viability_manifest, min_target_information=0.05)
    assert [fork.fork_id for fork in targets[3]] == ["informative"]
    quantile_targets = load_routing_targets(
        graph_manifest, viability_manifest, target_information_quantile=0.5
    )
    assert [fork.fork_id for fork in quantile_targets[3]] == ["informative"]
    weighted_targets = load_routing_targets(
        graph_manifest, viability_manifest, information_weighting=True
    )
    assert [fork.fork_id for fork in weighted_targets[3]] == ["informative"]
    assert 0.0 < weighted_targets[3][0].information_weight < 1.0


def test_beta_posterior_smooths_small_sample_action_targets(tmp_path: Path) -> None:
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
    viability_cache.write_text(json.dumps({
        "example_index": 3, "graph_sha256": "graph-hash",
        "forced_prefix_protocol": ASSISTANT_ACTION_PREFIX_PROTOCOL,
        "samples_per_action": 2, "temperature": 1.0,
        "fork_targets": [{"fork_id": "f", "action_ids": ["a", "b"],
                          "viability": {"a": 1.0, "b": 0.0},
                          "target": [0.7310585786, 0.2689414214]}],
    }) + "\n", encoding="utf-8")
    viability_manifest = tmp_path / "viability-manifest.json"
    viability_manifest.write_text(json.dumps({"schema_version": 1, "graph_cache_sha256": digest, "viability_cache": "viability.jsonl", "viability_cache_sha256": hashlib.sha256(viability_cache.read_bytes()).hexdigest(), "examples": 1, "forced_prefix_protocol": ASSISTANT_ACTION_PREFIX_PROTOCOL}), encoding="utf-8")

    target = load_routing_targets(graph_manifest, viability_manifest, viability_beta_prior=1.0)[3][0]
    # With two trials, Beta(1, 1) turns 1/0 into posterior means .75/.25;
    # its softmax target is less overconfident than the raw 1/0 target.
    assert 0.5 < target.actions[0].target_probability < 0.7310585786
    assert abs(sum(action.target_probability for action in target.actions) - 1.0) < 1e-8
