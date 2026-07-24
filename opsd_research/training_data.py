from __future__ import annotations

from datasets import Dataset, DatasetDict
from huggingface_hub import hf_hub_download
from pyarrow import parquet


DATASET_ID = "jasonrqh/Math-CoT-20k"
DATASET_FILE = "Math-CoT-20k.parquet"
DATASET_REVISION = "1435fb21d4fecc8ad4966a26f22a874cf2b527f1"


def load_math_cot_20k(revision: str = DATASET_REVISION) -> DatasetDict:
    """Load pinned rows while tolerating newer HF feature metadata.

    The Parquet schema uses the post-datasets-3.6 `List` feature label. OPSD's
    official environment pins datasets 3.6, so we read the same immutable
    Parquet file with PyArrow and drop only its library-specific schema
    metadata before constructing the Dataset.
    """
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
    return DatasetDict({"train": dataset})
