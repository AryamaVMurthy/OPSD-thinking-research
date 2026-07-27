import hashlib
import json
from pathlib import Path

from opsd_research.fluid_teacher_context import fluid_teacher_contexts
from opsd_research.graf_actions import ASSISTANT_ACTION_PREFIX_PROTOCOL


def test_fluid_context_appends_uncertain_action_outcomes_without_dropping_rows(
    tmp_path: Path,
) -> None:
    dossier_cache = tmp_path / "dossiers.jsonl"
    dossier_cache.write_text(
        "".join(
            json.dumps(
                {
                    "schema_version": 1,
                    "accepted": True,
                    "example_index": index,
                    "teacher_dossier": f"FREE FORM AUDIT {index}",
                    "blind_attempt_count": 2,
                }
            )
            + "\n"
            for index in (0, 1)
        ),
        encoding="utf-8",
    )
    dossier_manifest = tmp_path / "dossier-manifest.json"
    dossier_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cache": "dossiers.jsonl",
                "cache_sha256": hashlib.sha256(
                    dossier_cache.read_bytes()
                ).hexdigest(),
                "requested_examples": 2,
                "accepted_examples": 2,
                "rejected_examples": 0,
                "blind_attempts_per_problem": 2,
                "audit_format": "natural_language_v1",
                "schema_based_selection": False,
                "student_answer_context": False,
                "teacher_reference_context": True,
            }
        ),
        encoding="utf-8",
    )
    graph_cache = tmp_path / "graphs.jsonl"
    graph_cache.write_text(
        json.dumps(
            {
                "accepted": True,
                "example_index": 0,
                "graph_sha256": "graph-zero",
                "graph": {
                    "forks": [
                        {
                            "fork_id": "fork",
                            "state": "after expanding the product",
                            "actions": [
                                {
                                    "action_id": "a",
                                    "description": "factor the expression",
                                    "status": "viable",
                                },
                                {
                                    "action_id": "b",
                                    "description": "guess from small cases",
                                    "status": "risky",
                                },
                            ],
                        }
                    ]
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    graph_manifest = tmp_path / "graph-manifest.json"
    graph_digest = hashlib.sha256(graph_cache.read_bytes()).hexdigest()
    graph_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cache": "graphs.jsonl",
                "cache_sha256": graph_digest,
                "requested_examples": 1,
                "accepted_examples": 1,
                "rejected_examples": 0,
            }
        ),
        encoding="utf-8",
    )
    viability_cache = tmp_path / "viability.jsonl"
    viability_cache.write_text(
        json.dumps(
            {
                "example_index": 0,
                "graph_sha256": "graph-zero",
                "forced_prefix_protocol": ASSISTANT_ACTION_PREFIX_PROTOCOL,
                "samples_per_action": 2,
                "temperature": 1.0,
                "fork_targets": [
                    {
                        "fork_id": "fork",
                        "action_ids": ["a", "b"],
                        "viability": {"a": 1.0, "b": 0.5},
                        "samples_by_action": {"a": 2, "b": 2},
                        "target": [0.6224593312, 0.3775406688],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    viability_manifest = tmp_path / "viability-manifest.json"
    viability_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "graph_cache_sha256": graph_digest,
                "viability_cache": "viability.jsonl",
                "viability_cache_sha256": hashlib.sha256(
                    viability_cache.read_bytes()
                ).hexdigest(),
                "examples": 1,
                "forced_prefix_protocol": ASSISTANT_ACTION_PREFIX_PROTOCOL,
            }
        ),
        encoding="utf-8",
    )

    contexts = fluid_teacher_contexts(
        dossier_manifest, graph_manifest, viability_manifest
    )

    assert set(contexts) == {0, 1}
    assert contexts[1] == "FREE FORM AUDIT 1"
    assert "FREE FORM AUDIT 0" in contexts[0]
    assert "after expanding the product" in contexts[0]
    assert "factor the expression" in contexts[0]
    assert "2 of 2" in contexts[0]
    assert "1 of 2" in contexts[0]
    assert "small-sample evidence" in contexts[0]
