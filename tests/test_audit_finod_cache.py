from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from opsd_research.answer_masking import (
    ANSWER_LEAKAGE_PROTOCOL,
    PROBLEM_ONLY_GUIDANCE_PROTOCOL,
)
from opsd_research.audit_finod_cache import audit_finod_cache


def _cache(
    root: Path,
    question: str,
    *,
    validation_test: str = "Substitute into the original relation.",
) -> Path:
    graph = {
        "schema_version": 1,
        "problem_sha256": hashlib.sha256(question.encode()).hexdigest(),
        "forks": [
            {
                "fork_id": "f1",
                "state": "Choose a symbolic route.",
                "actions": [
                    {
                        "action_id": "a1",
                        "description": "Factor the symbolic expression.",
                        "status": "viable",
                        "validation_test": validation_test,
                        "recovery_action": "Return to the unsimplified relation.",
                    },
                    {
                        "action_id": "a2",
                        "description": "Use a discriminant argument.",
                        "status": "conditionally_viable",
                        "validation_test": "Check all domain restrictions.",
                        "recovery_action": "Compare against the factorization route.",
                    },
                ],
            }
        ],
    }
    record = {
        "schema_version": 1,
        "example_index": 0,
        "problem_sha256": hashlib.sha256(question.encode()).hexdigest(),
        "accepted": True,
        "guidance_input_protocol": PROBLEM_ONLY_GUIDANCE_PROTOCOL,
        "answer_leakage_protocol": ANSWER_LEAKAGE_PROTOCOL,
        "graph": graph,
        "graph_sha256": hashlib.sha256(
            json.dumps(
                graph, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest(),
    }
    cache = root / "graphs.jsonl"
    cache.write_text(json.dumps(record) + "\n", encoding="utf-8")
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cache": str(cache),
                "cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
                "requested_examples": 1,
                "accepted_examples": 1,
                "rejected_examples": 0,
                "training_dataset_revision": (
                    "1435fb21d4fecc8ad4966a26f22a874cf2b527f1"
                ),
                "guidance_input_protocol": PROBLEM_ONLY_GUIDANCE_PROTOCOL,
                "answer_leakage_protocol": ANSWER_LEAKAGE_PROTOCOL,
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_audit_replays_every_accepted_scaffold_through_training_checks(
    tmp_path: Path,
) -> None:
    question = "Find the requested integer."
    rows = [
        {
            "question": question,
            "response": r"A private derivation ends with \boxed{73}.",
            "data_source": "olympiads",
        }
    ]

    summary = audit_finod_cache(_cache(tmp_path, question), rows)

    assert summary["accepted_examples_replayed"] == 1
    assert summary["training_safe"] is True
    assert summary["guidance_input_protocol"] == "problem-only-v1"
    assert summary["accepted_by_data_source"] == {"olympiads": 1}


def test_audit_rejects_a_cached_numerical_result_before_publication(
    tmp_path: Path,
) -> None:
    question = "Find the requested integer."
    rows = [
        {
            "question": question,
            "response": r"A private derivation ends with \boxed{73}.",
            "data_source": "olympiads",
        }
    ]

    with pytest.raises(
        ValueError,
        match=r"source index 0.*numerical result",
    ):
        audit_finod_cache(
            _cache(
                tmp_path,
                question,
                validation_test="Verify that the intermediate product is 8.",
            ),
            rows,
        )
