import json

import pytest

from opsd_research import analyze_finod_guide_specificity as specificity


def _graph_record(example_index: int) -> dict:
    return {
        "accepted": True,
        "example_index": example_index,
        "graph": {
            "problem_sha256": "0" * 64,
            "schema_version": 1,
            "forks": [
                {
                    "fork_id": "fork_001",
                    "state": "initial",
                    "actions": [
                        {
                            "action_id": "action_001",
                            "description": "Use a direct coordinate representation.",
                            "validation_test": "Substitute into every constraint.",
                            "recovery_action": "Try an invariant if substitution fails.",
                            "status": "viable",
                        }
                    ],
                }
            ],
        },
    }


def test_load_samples_resolves_duplicate_question_to_guided_row(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    rows = [
        {"question": "Same problem", "response": r"\boxed{1}"},
        {"question": "Same problem", "response": r"\boxed{1}"},
    ]
    monkeypatch.setattr(
        specificity,
        "load_math_cot_20k",
        lambda heldout_fraction: {"train": rows},
    )
    graphs_path = tmp_path / "graphs.jsonl"
    graphs_path.write_text(
        json.dumps(_graph_record(example_index=0)) + "\n",
        encoding="utf-8",
    )
    generations_dir = tmp_path / "generations"
    generations_dir.mkdir()
    (generations_dir / "generations_step_1.json").write_text(
        json.dumps(
            {
                "generations": [
                    {
                        "prompt": (
                            "Problem: Same problem\n\n"
                            "Please reason step by step, and put your final answer"
                        ),
                        "completion": "No final box yet.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    samples = specificity.load_audit_samples(
        graphs_jsonl=graphs_path,
        generations_dir=generations_dir,
        samples_per_group=1,
        categories=("unfinished",),
    )

    assert samples[0].source_index == 0
    assert "coordinate representation" in samples[0].guide

