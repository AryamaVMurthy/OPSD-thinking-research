"""Answer-separated dataset construction for FiNOD."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Collection, Iterable, Sequence
from pathlib import Path
from typing import Any

from .generation_common import extract_last_boxed


OFFICIAL_HARDCODED_DATASET = "siyanzhao/Openthoughts_math_30k_opsd"
REPRESENTATIVE_SELECTION_PROTOCOL = (
    "data-source-response-length-quartile-v1"
)


def _row_response_length(row: dict[str, Any]) -> int:
    value = row.get("response_length")
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0
    ):
        return int(value)
    return len(str(row.get("response", "")))


def _bounded_proportional_allocation(
    counts: dict[tuple[str, int], int],
    *,
    limit: int,
) -> dict[tuple[str, int], int]:
    """Allocate an exact sample while retaining every populated stratum."""
    if not counts or any(count <= 0 for count in counts.values()):
        raise ValueError("representative strata must be nonempty")
    if limit < len(counts):
        raise ValueError(
            "representative limit is too small to cover every source-length "
            "stratum"
        )
    population = sum(counts.values())
    if limit > population:
        raise ValueError("representative limit exceeds eligible population")
    targets = {
        key: limit * count / population for key, count in counts.items()
    }
    allocation = {
        key: min(counts[key], max(1, math.floor(targets[key])))
        for key in counts
    }
    while sum(allocation.values()) < limit:
        candidates = [
            key for key in counts if allocation[key] < counts[key]
        ]
        if not candidates:
            raise RuntimeError("representative allocation exhausted capacity")
        key = max(
            candidates,
            key=lambda candidate: (
                targets[candidate] - allocation[candidate],
                counts[candidate] - allocation[candidate],
                candidate,
            ),
        )
        allocation[key] += 1
    while sum(allocation.values()) > limit:
        candidates = [
            key for key in counts if allocation[key] > 1
        ]
        if not candidates:
            raise RuntimeError("representative allocation cannot retain strata")
        key = max(
            candidates,
            key=lambda candidate: (
                allocation[candidate] - targets[candidate],
                allocation[candidate],
                candidate,
            ),
        )
        allocation[key] -= 1
    return allocation


def select_representative_finod_indices(
    rows: Sequence[dict[str, Any]],
    *,
    eligible_indices: Iterable[int],
    limit: int,
    seed: int,
) -> tuple[list[int], dict[str, Any]]:
    """Select an exact source- and length-stratified FiNOD training subset.

    ``data_source`` is the dataset's native seven-family provenance field.
    Within each source, eligible rows are split by response-length rank into
    quartiles. Content hashes choose rows inside each populated stratum, so
    selection does not depend on cache order or Python hash randomization.
    """
    eligible = sorted({int(index) for index in eligible_indices})
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ValueError("representative limit must be a positive integer")
    if len(eligible) < limit:
        raise ValueError(
            f"representative selection has {len(eligible)} eligible rows, "
            f"fewer than requested {limit}"
        )
    if any(index < 0 or index >= len(rows) for index in eligible):
        raise ValueError("representative eligible index is outside the dataset")

    by_source: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for index in eligible:
        row = rows[index]
        source = str(row.get("data_source") or "unknown")
        by_source[source].append((_row_response_length(row), index))

    strata: dict[tuple[str, int], list[int]] = defaultdict(list)
    for source, entries in sorted(by_source.items()):
        ranked = sorted(entries)
        count = len(ranked)
        for rank, (_length, index) in enumerate(ranked):
            quartile = min(3, rank * 4 // count)
            strata[(source, quartile)].append(index)
    allocation = _bounded_proportional_allocation(
        {key: len(indices) for key, indices in strata.items()},
        limit=limit,
    )

    selected: list[int] = []
    for key, indices in sorted(strata.items()):
        ranked = []
        for index in indices:
            row = rows[index]
            payload = json.dumps(
                {
                    "question": str(row.get("question", "")),
                    "selection_seed": int(seed),
                    "source_index": index,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            ranked.append(
                (hashlib.sha256(payload.encode("utf-8")).hexdigest(), index)
            )
        selected.extend(
            index
            for _digest, index in sorted(ranked)[: allocation[key]]
        )
    selected.sort()
    if len(selected) != limit or len(set(selected)) != limit:
        raise RuntimeError("representative selection did not produce exact rows")

    eligible_sources = Counter(
        str(rows[index].get("data_source") or "unknown")
        for index in eligible
    )
    selected_sources = Counter(
        str(rows[index].get("data_source") or "unknown")
        for index in selected
    )
    selected_strata = {
        f"{source}:q{quartile + 1}": allocation[(source, quartile)]
        for source, quartile in sorted(allocation)
    }
    indices_payload = json.dumps(
        selected, separators=(",", ":")
    ).encode("utf-8")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "selection_protocol": REPRESENTATIVE_SELECTION_PROTOCOL,
        "selection_seed": int(seed),
        "eligible_examples": len(eligible),
        "selected_examples": len(selected),
        "eligible_by_data_source": dict(sorted(eligible_sources.items())),
        "selected_by_data_source": dict(sorted(selected_sources.items())),
        "selected_by_length_stratum": selected_strata,
        "selected_indices": selected,
        "selected_indices_sha256": hashlib.sha256(indices_payload).hexdigest(),
    }
    return selected, manifest


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
