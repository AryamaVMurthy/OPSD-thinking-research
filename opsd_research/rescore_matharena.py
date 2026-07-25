from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml

from .config import MATH_DATASETS, load_config
from .matharena_grading import rescore_record, summarize_rescored
from .records import read_jsonl, validate_consistent_fields


COMPETITION_CONFIGS = {
    "aime24": (
        "configs/competitions/aime/aime_2024_I.yaml",
        "configs/competitions/aime/aime_2024_II.yaml",
    ),
    "aime25": "configs/competitions/aime/aime_2025.yaml",
    "aime26": "configs/competitions/aime/aime_2026.yaml",
    "hmmt25": "configs/competitions/hmmt/hmmt_feb_2025.yaml",
}


def _matharena_root() -> Path:
    return Path(__file__).resolve().parents[1] / "third_party" / "matharena"


def _matharena_revision() -> str:
    root = _matharena_root()
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    revision = result.stdout.strip()
    if len(revision) != 40 or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise RuntimeError(f"invalid MathArena revision: {revision!r}")
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("official MathArena checkout has tracked local changes")
    return revision


def _competition_protocol(dataset: str) -> tuple[bool, str, str]:
    try:
        relative_path = COMPETITION_CONFIGS[dataset]
    except KeyError as exc:
        raise ValueError(f"no official MathArena config mapped for {dataset!r}") from exc
    relative_paths = (relative_path,) if isinstance(relative_path, str) else relative_path
    payloads = []
    strict_values = []
    for item in relative_paths:
        path = _matharena_root() / item
        payload = path.read_bytes()
        protocol = yaml.safe_load(payload)
        if not isinstance(protocol, dict):
            raise ValueError(f"{path}: expected YAML mapping")
        payloads.append(payload)
        strict_values.append(bool(protocol.get("strict_parsing", False)))
    if len(set(strict_values)) != 1:
        raise ValueError(f"{dataset}: component protocols disagree on strict parsing")
    return (
        strict_values[0],
        "+".join(relative_paths),
        hashlib.sha256(b"\0".join(payloads)).hexdigest(),
    )


def _scorer_environment() -> dict[str, str]:
    packages = (
        "antlr4-python3-runtime",
        "loguru",
        "PyYAML",
        "regex",
        "sympy",
    )
    return {
        "python": platform.python_version(),
        **{package: importlib.metadata.version(package) for package in packages},
    }


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def rescore_run(
    *,
    config_path: Path,
    input_paths: list[Path],
    grades_output: Path,
    summary_output: Path,
) -> dict[str, Any]:
    config = load_config(config_path).data
    if config["kind"] != "math_eval":
        raise ValueError("official MathArena rescoring requires kind=math_eval")
    records = [record for path in input_paths for record in read_jsonl(path)]
    identity = validate_consistent_fields(
        records,
        (
            "model",
            "model_revision",
            "method",
            "checkpoint",
            "benchmark",
            "dataset_revision",
            "adapter_sha256",
            "seed_protocol",
        ),
    )
    for field in ("model", "model_revision", "dataset_revision"):
        if identity[field] != config[field]:
            raise ValueError(
                f"generation {field}={identity[field]!r} does not match "
                f"config value {config[field]!r}"
            )
    if identity["benchmark"] != config["dataset"]:
        raise ValueError(
            f"generation benchmark={identity['benchmark']!r} does not match "
            f"config dataset={config['dataset']!r}"
        )
    expected_problems = MATH_DATASETS[config["dataset"]]["expected_count"]
    problem_ids = {str(record["problem_id"]) for record in records}
    if len(problem_ids) != expected_problems:
        raise ValueError(
            f"expected {expected_problems} problems, found {len(problem_ids)}"
        )

    strict_parsing, competition_config, competition_config_sha256 = (
        _competition_protocol(config["dataset"])
    )
    grader_revision = _matharena_revision()
    sidecars = [
        rescore_record(record, strict_parsing=strict_parsing) for record in records
    ]
    summary = summarize_rescored(
        records,
        sidecars,
        samples_per_problem=int(config["samples_per_problem"]),
        grader_revision=grader_revision,
        strict_parsing=strict_parsing,
    )
    summary["competition_config"] = competition_config
    summary["competition_config_sha256"] = competition_config_sha256
    summary["scorer_environment"] = _scorer_environment()
    summary["raw_inputs"] = [
        {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in input_paths
    ]

    grades_payload = "".join(
        json.dumps(sidecar, ensure_ascii=False, sort_keys=True) + "\n"
        for sidecar in sidecars
    )
    _atomic_write(grades_output, grades_payload)
    _atomic_write(
        summary_output,
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rescore immutable math generations with pinned MathArena."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--input", required=True, nargs="+", type=Path)
    parser.add_argument("--grades-output", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = rescore_run(
        config_path=args.config,
        input_paths=args.input,
        grades_output=args.grades_output,
        summary_output=args.summary_output,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
