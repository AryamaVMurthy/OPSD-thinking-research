"""Teacher-only dataset redirect for contrastive-hindsight OPSD."""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path

from .ch_dossier import accepted_teacher_dossiers, select_teacher_dossiers


OFFICIAL_HARDCODED_DATASET = "siyanzhao/Openthoughts_math_30k_opsd"


def install_ch_dossier_dataset_redirect(
    manifest_path: str | Path,
    *,
    source_indices: Collection[int] | None = None,
) -> int:
    """Attach dossiers only to OPSD's privileged ``solution`` field."""
    return _install_dossier_dataset_redirect(
        manifest_path,
        source_indices=source_indices,
        include_graf_source_index=False,
    )


def install_fluid_dossier_dataset_redirect(
    manifest_path: str | Path,
    *,
    source_indices: Collection[int] | None = None,
) -> int:
    """Keep all dossier rows in base OPSD while enabling sparse GRAF joins.

    Every selected identity receives its natural teacher-only dossier.  The
    immutable source index is metadata consumed by the auxiliary loss, never
    student-visible text.  Identities without a reliable routed target simply
    receive the ordinary OPSD update.
    """
    return _install_dossier_dataset_redirect(
        manifest_path,
        source_indices=source_indices,
        include_graf_source_index=True,
    )


def _install_dossier_dataset_redirect(
    manifest_path: str | Path,
    *,
    source_indices: Collection[int] | None,
    include_graf_source_index: bool,
) -> int:
    dossiers = select_teacher_dossiers(
        accepted_teacher_dossiers(manifest_path), source_indices
    )
    import datasets

    original_load_dataset = datasets.load_dataset

    def pinned_load_dataset(path, *args, **kwargs):
        if path != OFFICIAL_HARDCODED_DATASET:
            return original_load_dataset(path, *args, **kwargs)
        if args or kwargs:
            raise RuntimeError(
                "official OPSD dataset call unexpectedly supplied arguments"
            )
        from .training_data import load_math_cot_20k

        loaded = load_math_cot_20k(heldout_fraction=0.0)["train"]
        selected_indices = sorted(dossiers)
        selected = loaded.select(selected_indices).add_column(
            "_ch_source_index", selected_indices
        )

        def normalize(example):
            normalized = {
                # The upstream collator gives only this field to the student.
                "problem": example["question"],
                # The upstream collator gives this field only to the teacher.
                "solution": dossiers[int(example["_ch_source_index"])],
            }
            if include_graf_source_index:
                normalized["graf_source_index"] = int(
                    example["_ch_source_index"]
                )
            return normalized

        return datasets.DatasetDict(
            {
                "train": selected.map(
                    normalize,
                    remove_columns=selected.column_names,
                    desc="Attaching verified teacher-only CH dossiers",
                )
            }
        )

    datasets.load_dataset = pinned_load_dataset
    return len(dossiers)
