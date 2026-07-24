from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

from .config import ALLOWED_MODELS


PINNED_DATASET = "jasonrqh/Math-CoT-20k"
OFFICIAL_HARDCODED_DATASET = "siyanzhao/Openthoughts_math_30k_opsd"


def _argument(name: str) -> str | None:
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except (ValueError, IndexError):
        return None


def _flag(name: str) -> bool:
    return name in sys.argv


def _validate_invocation() -> None:
    model = _argument("--model_name_or_path")
    revision = _argument("--model_revision")
    student_revision = _argument("--student_model_revision")
    max_steps = _argument("--max_steps")
    if model not in ALLOWED_MODELS:
        raise SystemExit(f"forbidden model {model!r}; expected an instruction Qwen3 checkpoint")
    if model.endswith("-Base"):
        raise SystemExit("Qwen3 Base checkpoints are forbidden")
    if not revision or revision in {"main", "latest"}:
        raise SystemExit("--model_revision must be an immutable commit")
    if student_revision != revision:
        raise SystemExit(
            "--student_model_revision must match --model_revision so vLLM rollouts "
            "use the pinned training checkpoint"
        )
    if max_steps != "200":
        raise SystemExit("--max_steps must be exactly 200")
    for flag in ("--student_thinking", "--teacher_thinking", "--fixed_teacher", "--use_peft"):
        if not _flag(flag):
            raise SystemExit(f"required flag is missing: {flag}")


def _install_dataset_redirect() -> None:
    from .training_data import load_math_cot_20k

    revision = os.environ.get("OPSD_DATASET_REVISION")
    if not revision or revision in {"main", "latest"}:
        raise SystemExit("OPSD_DATASET_REVISION must be an immutable commit")

    import datasets

    original_load_dataset = datasets.load_dataset

    def pinned_load_dataset(path, *args, **kwargs):
        if path != OFFICIAL_HARDCODED_DATASET:
            return original_load_dataset(path, *args, **kwargs)
        if args or kwargs:
            raise RuntimeError(
                "official OPSD dataset call unexpectedly supplied arguments; "
                "refusing an ambiguous redirect"
            )
        loaded = load_math_cot_20k(revision)

        def normalize(example):
            return {
                "problem": example["question"],
                "solution": example["response"],
            }

        return loaded.map(
            normalize,
            remove_columns=loaded["train"].column_names,
            desc="Normalizing pinned Math-CoT-20k for official OPSD",
        )

    datasets.load_dataset = pinned_load_dataset


def main() -> None:
    _validate_invocation()
    _install_dataset_redirect()
    upstream = Path(__file__).resolve().parents[1] / "third_party" / "opsd"
    sys.path.insert(0, str(upstream))
    runpy.run_path(str(upstream / "opsd_train.py"), run_name="__main__")


if __name__ == "__main__":
    main()
