from __future__ import annotations

import sys
from pathlib import Path


def upstream_path() -> Path:
    return Path(__file__).resolve().parents[1] / "third_party" / "livecodebench"


def load_lcb_v6(revision: str):
    """Load the immutable v6 slice and reuse LiveCodeBench's official schema."""
    from datasets import load_dataset

    sys.path.insert(0, str(upstream_path()))
    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem

    dataset = load_dataset(
        "livecodebench/code_generation_lite",
        split="test",
        version_tag="v6",
        revision=revision,
        trust_remote_code=True,
    )
    return sorted(
        (CodeGenerationProblem(**row) for row in dataset),
        key=lambda item: item.question_id,
    )
