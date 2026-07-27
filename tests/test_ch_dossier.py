import hashlib
import json
from pathlib import Path
import sys
import types

import pytest

from opsd_research.ch_dossier_dataset import (
    OFFICIAL_HARDCODED_DATASET,
    install_ch_dossier_dataset_redirect,
    install_fluid_dossier_dataset_redirect,
)
from opsd_research.ch_dossier import (
    accepted_teacher_dossiers,
    select_teacher_dossiers,
)


def test_only_hash_verified_accepted_dossiers_are_loaded(tmp_path: Path) -> None:
    cache = tmp_path / "dossiers.jsonl"
    accepted = {
        "schema_version": 1,
        "accepted": True,
        "example_index": 11,
        "problem_sha256": "problem-hash",
        "teacher_dossier": "privileged corrected context",
        "blind_attempt_count": 3,
    }
    rejected = {
        "schema_version": 1,
        "accepted": False,
        "example_index": 12,
        "problem_sha256": "other-hash",
        "rejection_reason": "auditor response was empty",
    }
    cache.write_text(
        json.dumps(accepted) + "\n" + json.dumps(rejected) + "\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cache": "dossiers.jsonl",
                "cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
                "requested_examples": 2,
                "accepted_examples": 1,
                "rejected_examples": 1,
                "blind_attempts_per_problem": 3,
                "audit_format": "natural_language_v1",
                "schema_based_selection": False,
                "student_answer_context": False,
                "teacher_reference_context": True,
            }
        ),
        encoding="utf-8",
    )

    assert accepted_teacher_dossiers(manifest) == {
        11: "privileged corrected context"
    }


def test_teacher_dossier_loader_rejects_schema_selected_cache(tmp_path: Path) -> None:
    cache = tmp_path / "dossiers.jsonl"
    cache.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "accepted": True,
                "example_index": 3,
                "teacher_dossier": "free-form natural audit",
                "blind_attempt_count": 3,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cache": "dossiers.jsonl",
                "cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
                "requested_examples": 1,
                "accepted_examples": 1,
                "rejected_examples": 0,
                "blind_attempts_per_problem": 3,
                "audit_format": "natural_language_v1",
                "schema_based_selection": True,
                "student_answer_context": False,
                "teacher_reference_context": True,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schema-based"):
        accepted_teacher_dossiers(manifest)


def test_teacher_dossier_selection_is_exact_and_nonempty() -> None:
    dossiers = {7: "seven", 2: "two", 5: "five"}
    assert select_teacher_dossiers(dossiers, [7, 2]) == {2: "two", 7: "seven"}
    with pytest.raises(ValueError, match="not accepted dossier rows"):
        select_teacher_dossiers(dossiers, [9])
    with pytest.raises(ValueError, match="must not be empty"):
        select_teacher_dossiers(dossiers, [])


def test_dataset_redirect_keeps_privileged_context_out_of_student_problem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "dossiers.jsonl"
    cache.write_text(json.dumps({
        "schema_version": 1, "accepted": True, "example_index": 0,
        "teacher_dossier": "SECRET REFERENCE AND AUDIT",
        "blind_attempt_count": 3,
    }) + "\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": 1, "cache": str(cache),
        "cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
        "requested_examples": 1, "accepted_examples": 1,
        "rejected_examples": 0, "blind_attempts_per_problem": 3,
        "audit_format": "natural_language_v1",
        "schema_based_selection": False,
        "student_answer_context": False, "teacher_reference_context": True,
    }), encoding="utf-8")

    class FakeDataset(list):
        @property
        def column_names(self):
            return list(self[0]) if self else []

        def select(self, indices):
            return FakeDataset([dict(self[index]) for index in indices])

        def add_column(self, name, values):
            return FakeDataset([
                {**row, name: value} for row, value in zip(self, values, strict=True)
            ])

        def map(self, function, **_kwargs):
            return FakeDataset([function(row) for row in self])

    fake_datasets = types.SimpleNamespace(
        load_dataset=lambda *_args, **_kwargs: "original",
        DatasetDict=dict,
    )
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    monkeypatch.setattr(
        "opsd_research.training_data.load_math_cot_20k",
        lambda **_kwargs: {
            "train": FakeDataset([
                {"question": "PUBLIC PROBLEM", "response": "source solution"}
            ])
        },
    )

    assert install_ch_dossier_dataset_redirect(manifest) == 1
    row = fake_datasets.load_dataset(OFFICIAL_HARDCODED_DATASET)["train"][0]
    assert row == {
        "problem": "PUBLIC PROBLEM",
        "solution": "SECRET REFERENCE AND AUDIT",
    }


def test_fluid_redirect_preserves_every_dossier_identity_for_base_opsd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "dossiers.jsonl"
    records = [
        {
            "schema_version": 1,
            "accepted": True,
            "example_index": index,
            "teacher_dossier": f"teacher-only-{index}",
            "blind_attempt_count": 3,
        }
        for index in (0, 1)
    ]
    cache.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cache": str(cache),
                "cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
                "requested_examples": 2,
                "accepted_examples": 2,
                "rejected_examples": 0,
                "blind_attempts_per_problem": 3,
                "audit_format": "natural_language_v1",
                "schema_based_selection": False,
                "student_answer_context": False,
                "teacher_reference_context": True,
            }
        ),
        encoding="utf-8",
    )

    class FakeDataset(list):
        @property
        def column_names(self):
            return list(self[0]) if self else []

        def select(self, indices):
            return FakeDataset([dict(self[index]) for index in indices])

        def add_column(self, name, values):
            return FakeDataset(
                [
                    {**row, name: value}
                    for row, value in zip(self, values, strict=True)
                ]
            )

        def map(self, function, **_kwargs):
            return FakeDataset([function(row) for row in self])

    fake_datasets = types.SimpleNamespace(
        load_dataset=lambda *_args, **_kwargs: "original",
        DatasetDict=dict,
    )
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    monkeypatch.setattr(
        "opsd_research.training_data.load_math_cot_20k",
        lambda **_kwargs: {
            "train": FakeDataset(
                [
                    {"question": "PUBLIC ZERO", "response": "reference zero"},
                    {"question": "PUBLIC ONE", "response": "reference one"},
                ]
            )
        },
    )

    assert (
        install_fluid_dossier_dataset_redirect(
            manifest,
            teacher_contexts={
                0: "ENRICHED TEACHER ZERO",
                1: "ENRICHED TEACHER ONE",
            },
        )
        == 2
    )
    rows = fake_datasets.load_dataset(OFFICIAL_HARDCODED_DATASET)["train"]
    assert rows == [
        {
            "problem": "PUBLIC ZERO",
            "solution": "ENRICHED TEACHER ZERO",
            "graf_source_index": 0,
        },
        {
            "problem": "PUBLIC ONE",
            "solution": "ENRICHED TEACHER ONE",
            "graf_source_index": 1,
        },
    ]


def test_fluid_redirect_rejects_partial_teacher_contexts(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "dossiers.jsonl"
    cache.write_text(
        "".join(
            json.dumps(
                {
                    "schema_version": 1,
                    "accepted": True,
                    "example_index": index,
                    "teacher_dossier": f"teacher-only-{index}",
                    "blind_attempt_count": 2,
                }
            )
            + "\n"
            for index in (0, 1)
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cache": str(cache),
                "cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
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

    with pytest.raises(
        ValueError,
        match="fluid teacher contexts must cover every selected dossier",
    ):
        install_fluid_dossier_dataset_redirect(
            manifest,
            teacher_contexts={0: "ONLY ZERO"},
        )
