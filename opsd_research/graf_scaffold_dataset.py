"""Pinned answer-masked graph scaffold dataset redirect for GRAF training."""

from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Collection
from typing import Any

from .graf_cache import validate_graph_cache_manifest
from .graf_graph import GraphAction, GraphFork, ReasoningGraph, render_graph_scaffold


OFFICIAL_HARDCODED_DATASET = "siyanzhao/Openthoughts_math_30k_opsd"


def _graph(payload: dict[str, Any]) -> ReasoningGraph:
    return ReasoningGraph(
        problem_sha256=str(payload["problem_sha256"]),
        forks=tuple(
            GraphFork(
                fork_id=str(fork["fork_id"]),
                state=str(fork["state"]),
                actions=tuple(GraphAction(**action) for action in fork["actions"]),
            )
            for fork in payload["forks"]
        ),
        schema_version=int(payload.get("schema_version", 1)),
    )


def accepted_scaffolds(manifest_path: str | Path) -> dict[int, str]:
    """Load only parsed accepted graph records; never use a builder response directly."""
    manifest = validate_graph_cache_manifest(manifest_path)
    cache_path = Path(str(manifest["cache"]))
    if not cache_path.is_absolute():
        cache_path = Path(manifest_path).parent / cache_path
    scaffolds: dict[int, str] = {}
    for line_number, line in enumerate(cache_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record: dict[str, Any] = json.loads(line)
        if not record.get("accepted"):
            continue
        index = int(record["example_index"])
        if index in scaffolds:
            raise ValueError(f"duplicate accepted graph example_index {index} at line {line_number}")
        scaffolds[index] = render_graph_scaffold(_graph(record["graph"]))
    if not scaffolds:
        raise ValueError("graph cache has no accepted answer-masked scaffolds")
    if len(scaffolds) != int(manifest["accepted_examples"]):
        raise ValueError("graph-cache accepted count does not match its manifest")
    return scaffolds


def select_scaffolds(
    scaffolds: dict[int, str], source_indices: Collection[int] | None
) -> dict[int, str]:
    """Return a deterministic, validated subset of immutable graph rows."""
    if source_indices is None:
        return scaffolds
    selected = {int(index) for index in source_indices}
    unknown = selected.difference(scaffolds)
    if unknown:
        raise ValueError(
            "requested GRAF source indices are not accepted graph rows: "
            f"{sorted(unknown)[:5]}"
        )
    if not selected:
        raise ValueError("GRAF source-index selection must not be empty")
    return {index: scaffolds[index] for index in sorted(selected)}


def install_graph_scaffold_dataset_redirect(
    manifest_path: str | Path,
    *,
    source_indices: Collection[int] | None = None,
) -> int:
    """Replace upstream's hard-coded dataset with verified graph-scaffold rows.

    ``source_indices`` is deliberately an immutable cache identity rather than
    a sampling hint.  A viability-routed run must use exactly the rows for
    which it has independently measured targets: otherwise many updates carry
    no routed branch signal and the effective objective weight depends on the
    data-loader shuffle.  Scaffold-only candidates retain the full accepted
    graph set by leaving this argument unset.
    """
    scaffolds = select_scaffolds(accepted_scaffolds(manifest_path), source_indices)
    import datasets

    original_load_dataset = datasets.load_dataset

    def pinned_load_dataset(path, *args, **kwargs):
        if path != OFFICIAL_HARDCODED_DATASET:
            return original_load_dataset(path, *args, **kwargs)
        if args or kwargs:
            raise RuntimeError("official OPSD dataset call unexpectedly supplied arguments")
        from .training_data import load_math_cot_20k
        # Cache identities always use the full source corpus. The launcher
        # selects training indices afterwards, so a held-out row cannot become
        # model-visible merely because it has a cached scaffold.
        loaded = load_math_cot_20k(heldout_fraction=0.0)["train"]
        selected_indices = sorted(scaffolds)
        selected = loaded.select(selected_indices).add_column(
            "_graf_source_index", selected_indices
        )

        def normalize(example):
            return {
                "problem": example["question"],
                "solution": scaffolds[int(example["_graf_source_index"])],
                # The index is immutable cache identity, not model-visible text.
                # It lets the loss join each minibatch item to its independently
                # verified forced-continuation targets.
                "graf_source_index": int(example["_graf_source_index"]),
            }

        return datasets.DatasetDict({
            "train": selected.map(
                normalize,
                remove_columns=selected.column_names,
                desc="Attaching verified answer-masked GRAF scaffolds",
            )
        })

    datasets.load_dataset = pinned_load_dataset
    return len(scaffolds)
