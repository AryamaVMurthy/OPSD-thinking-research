"""Answer-separated dataset construction for FiNOD."""

from __future__ import annotations

import re
from collections.abc import Collection
from pathlib import Path

from .generation_common import extract_last_boxed


OFFICIAL_HARDCODED_DATASET = "siyanzhao/Openthoughts_math_30k_opsd"


def remove_graph_identifiers(guide: str) -> str:
    """Remove cache-local IDs that can coincide with a short reference answer.

    Fork and action IDs carry no procedural content.  In particular, an action
    named ``4`` is not an answer leak, but placing that token in the teacher
    view would make a literal leakage audit indistinguishable from one.
    """
    cleaned = []
    for line in str(guide).splitlines():
        line = re.sub(r"^State\s+\S+:\s*", "Strategy state: ", line)
        line = re.sub(r"^-\s+\S+\s+(\[[^\]]+\]:)", r"- \1", line)
        cleaned.append(line)
    return "\n".join(cleaned)


def finod_training_row(
    *,
    question: str,
    reference_solution: str,
    answer_masked_guide: str,
    source_index: int,
) -> dict[str, object]:
    """Create one row with mutually explicit student, guide, and answer fields."""
    problem = str(question).strip()
    guide = str(answer_masked_guide).strip()
    answer = extract_last_boxed(str(reference_solution))
    if not problem:
        raise ValueError("FiNOD requires a nonempty problem")
    if not guide:
        raise ValueError("FiNOD requires nonempty procedural guidance")
    if not answer:
        raise ValueError("FiNOD reference solution lacks a boxed answer")
    if "\\boxed" in guide:
        raise ValueError("FiNOD guide contains an answer-format leak")
    if re.search(
        rf"(?<![A-Za-z0-9]){re.escape(answer)}(?![A-Za-z0-9])",
        guide,
    ):
        raise ValueError("FiNOD guide contains the reference answer")
    return {
        "problem": problem,
        "solution": guide,
        "finod_answer_control": answer,
        "finod_source_index": int(source_index),
    }


def install_finod_dataset_redirect(
    graph_manifest_path: str | Path,
    *,
    source_indices: Collection[int] | None = None,
) -> int:
    """Attach verified guidance and an isolated answer-control metadata field."""
    from .graf_scaffold_dataset import accepted_scaffolds, select_scaffolds

    scaffolds = select_scaffolds(
        accepted_scaffolds(graph_manifest_path), source_indices
    )
    import datasets

    original_load_dataset = datasets.load_dataset

    def pinned_load_dataset(path, *args, **kwargs):
        if path != OFFICIAL_HARDCODED_DATASET:
            return original_load_dataset(path, *args, **kwargs)
        if args or kwargs:
            raise RuntimeError(
                "upstream OPSD dataset call unexpectedly supplied arguments"
            )
        from .training_data import load_math_cot_20k

        raw = load_math_cot_20k(heldout_fraction=0.0)["train"]
        selected_indices = sorted(scaffolds)
        selected = raw.select(selected_indices).add_column(
            "_finod_source_index", selected_indices
        )

        def normalize(example):
            index = int(example["_finod_source_index"])
            return finod_training_row(
                question=example["question"],
                reference_solution=example["response"],
                answer_masked_guide=remove_graph_identifiers(scaffolds[index]),
                source_index=index,
            )

        return datasets.DatasetDict(
            {
                "train": selected.map(
                    normalize,
                    remove_columns=selected.column_names,
                    desc="Attaching FiNOD guide and isolated answer control",
                )
            }
        )

    datasets.load_dataset = pinned_load_dataset
    return len(scaffolds)
