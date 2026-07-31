from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from collections.abc import Sequence
from typing import Any

DATASET_ID = "jasonrqh/Math-CoT-20k"
DATASET_FILE = "Math-CoT-20k.parquet"
DATASET_REVISION = "1435fb21d4fecc8ad4966a26f22a874cf2b527f1"
HELDOUT_FRACTION_ENV = "OPSD_HELDOUT_DIAGNOSTIC_FRACTION"


def _validate_fraction(value: float) -> float:
    if not 0.0 <= value < 0.5:
        raise ValueError("heldout diagnostic fraction must be in [0, 0.5)")
    return value


def heldout_fraction_from_environment() -> float:
    raw = os.environ.get(HELDOUT_FRACTION_ENV, "0")
    try:
        return _validate_fraction(float(raw))
    except ValueError as error:
        raise ValueError(
            f"{HELDOUT_FRACTION_ENV} must be a number in [0, 0.5), got {raw!r}"
        ) from error


def _row_digest(row: dict[str, object]) -> str:
    """Return a stable content identity, independent of source row order."""
    payload = json.dumps(
        {"question": str(row["question"]), "response": str(row["response"])},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _problem_row_digest(row: dict[str, object]) -> str:
    """Return an identity using only public problem/provenance fields."""
    payload = json.dumps(
        {
            "question": str(row["question"]),
            "data_source": str(row.get("data_source") or "unknown"),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def partition_indices(
    dataset: Sequence[dict[str, object]], heldout_fraction: float
) -> tuple[list[int], list[int]]:
    """Return stable source indices for training and held-out diagnostics."""
    fraction = _validate_fraction(heldout_fraction)
    if fraction == 0.0:
        return list(range(len(dataset))), []
    threshold = int(fraction * (1 << 64))
    train, heldout = [], []
    for index, row in enumerate(dataset):
        value = int(_row_digest(row)[:16], 16)
        (heldout if value < threshold else train).append(index)
    if not train or not heldout:
        raise ValueError(
            "content-hash partition produced an empty split; choose a usable heldout fraction"
        )
    return train, heldout


def problem_partition_indices(
    dataset: Sequence[dict[str, object]], heldout_fraction: float
) -> tuple[list[int], list[int]]:
    """Partition rows without materializing an answer or reference response."""
    fraction = _validate_fraction(heldout_fraction)
    if fraction == 0.0:
        return list(range(len(dataset))), []
    threshold = int(fraction * (1 << 64))
    train, heldout = [], []
    for index, row in enumerate(dataset):
        value = int(_problem_row_digest(row)[:16], 16)
        (heldout if value < threshold else train).append(index)
    if not train or not heldout:
        raise ValueError(
            "problem-only hash partition produced an empty split; choose a "
            "usable heldout fraction"
        )
    return train, heldout


def partition_manifest(
    dataset: Sequence[dict[str, object]], heldout_fraction: float
) -> dict[str, object]:
    train, heldout = partition_indices(dataset, heldout_fraction)
    heldout_digests = [_row_digest(dataset[index]) for index in heldout]
    return {
        "schema_version": 1,
        "dataset": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "partition_protocol": "content-sha256-first64-threshold-v1",
        "heldout_diagnostic_fraction": heldout_fraction,
        "source_examples": len(dataset),
        "train_examples": len(train),
        "heldout_diagnostic_examples": len(heldout),
        "heldout_content_sha256": hashlib.sha256(
            "\n".join(heldout_digests).encode("utf-8")
        ).hexdigest(),
    }


def problem_partition_manifest(
    dataset: Sequence[dict[str, object]], heldout_fraction: float
) -> dict[str, object]:
    """Describe a split whose inputs contain no response/answer column."""
    train, heldout = problem_partition_indices(dataset, heldout_fraction)
    heldout_digests = [
        _problem_row_digest(dataset[index]) for index in heldout
    ]
    return {
        "schema_version": 1,
        "dataset": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "partition_protocol": (
            "problem-provenance-sha256-first64-threshold-v1"
        ),
        "heldout_diagnostic_fraction": heldout_fraction,
        "source_examples": len(dataset),
        "train_examples": len(train),
        "heldout_diagnostic_examples": len(heldout),
        "heldout_problem_sha256": hashlib.sha256(
            "\n".join(heldout_digests).encode("utf-8")
        ).hexdigest(),
        "answer_access": False,
        "reference_solution_access": False,
    }


def write_partition_manifest(
    path: Path, dataset: Sequence[dict[str, object]], heldout_fraction: float
) -> dict[str, object]:
    manifest = partition_manifest(dataset, heldout_fraction)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def write_problem_partition_manifest(
    path: Path,
    dataset: Sequence[dict[str, object]],
    heldout_fraction: float,
) -> dict[str, object]:
    manifest = problem_partition_manifest(dataset, heldout_fraction)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_math_cot_20k(
    revision: str = DATASET_REVISION,
    *,
    heldout_fraction: float | None = None,
) -> Any:
    """Load pinned rows while tolerating newer HF feature metadata.

    The Parquet schema uses the post-datasets-3.6 `List` feature label. OPSD's
    official environment pins datasets 3.6, so we read the same immutable
    Parquet file with PyArrow and drop only its library-specific schema
    metadata before constructing the Dataset.
    """
    from datasets import Dataset, DatasetDict
    from huggingface_hub import hf_hub_download
    from pyarrow import parquet

    path = hf_hub_download(
        repo_id=DATASET_ID,
        filename=DATASET_FILE,
        repo_type="dataset",
        revision=revision,
    )
    table = parquet.read_table(path).replace_schema_metadata(None)
    dataset = Dataset(table)
    required = {"question", "response"}
    if not required.issubset(dataset.column_names):
        raise RuntimeError(
            f"training dataset missing columns: {required - set(dataset.column_names)}"
        )
    fraction = (
        heldout_fraction_from_environment()
        if heldout_fraction is None
        else _validate_fraction(heldout_fraction)
    )
    train, heldout = partition_indices(dataset, fraction)
    return DatasetDict({
        "train": dataset.select(train),
        "heldout_diagnostic": dataset.select(heldout),
    })


def load_math_cot_questions_only(
    revision: str = DATASET_REVISION,
) -> Any:
    """Load only public problem/provenance columns from the pinned Parquet.

    PyArrow column projection happens while reading the file.  The response
    column is therefore never materialized in the guidance-builder process.
    """
    from datasets import Dataset
    from huggingface_hub import hf_hub_download
    from pyarrow import parquet

    path = hf_hub_download(
        repo_id=DATASET_ID,
        filename=DATASET_FILE,
        repo_type="dataset",
        revision=revision,
    )
    columns = ["question", "data_source"]
    table = parquet.read_table(path, columns=columns).replace_schema_metadata(
        None
    )
    dataset = Dataset(table)
    if dataset.column_names != columns:
        raise RuntimeError(
            "problem-only dataset projection returned unexpected columns"
        )
    return dataset
