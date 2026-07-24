from __future__ import annotations

from typing import Any


def configure_structured_dataset_args(args: Any) -> Any:
    """Keep OPSD's structured examples intact for its custom data collator."""
    dataset_kwargs = dict(getattr(args, "dataset_kwargs", None) or {})
    dataset_kwargs["skip_prepare_dataset"] = True
    args.dataset_kwargs = dataset_kwargs
    args.remove_unused_columns = False
    return args
