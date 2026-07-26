import hashlib
import json
from pathlib import Path

import pytest

from opsd_research.graf_scaffold_dataset import accepted_scaffolds, select_scaffolds


def test_loads_only_verified_accepted_graphs(tmp_path: Path) -> None:
    cache = tmp_path / "graphs.jsonl"
    record = {
        "accepted": True,
        "example_index": 4,
        "graph": {
            "problem_sha256": "abc",
            "forks": [{
                "fork_id": "f0", "state": "inspect structure",
                "actions": [{
                    "action_id": "a0", "description": "factor symbolically",
                    "status": "viable", "validation_test": "check factors",
                    "recovery_action": "try substitution",
                }, {
                    "action_id": "a1", "description": "use invariants",
                    "status": "risky", "validation_test": "check domain",
                    "recovery_action": "return to factors",
                }],
            }],
        },
    }
    cache.write_text(json.dumps(record) + "\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": 1, "cache": "graphs.jsonl",
        "cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
        "requested_examples": 1, "accepted_examples": 1, "rejected_examples": 0,
    }), encoding="utf-8")

    scaffolds = accepted_scaffolds(manifest)
    assert list(scaffolds) == [4]
    assert "factor symbolically" in scaffolds[4]


def test_select_scaffolds_is_exact_and_rejects_unverified_indices() -> None:
    scaffolds = {7: "seven", 2: "two", 5: "five"}
    assert select_scaffolds(scaffolds, [7, 2]) == {2: "two", 7: "seven"}
    with pytest.raises(ValueError, match="not accepted graph rows"):
        select_scaffolds(scaffolds, [9])
    with pytest.raises(ValueError, match="must not be empty"):
        select_scaffolds(scaffolds, [])
