from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


def stable_seed(
    base_seed: int,
    model: str,
    method: str,
    checkpoint: str,
    benchmark: str,
    problem_id: str,
    sample_index: int,
) -> int:
    payload = "\x1f".join(
        [
            str(base_seed),
            model,
            method,
            checkpoint,
            benchmark,
            problem_id,
            str(sample_index),
        ]
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def validate_adapter_identity(
    adapter: str | Path | None, adapter_sha256: str | None
) -> None:
    if bool(adapter) != bool(adapter_sha256):
        raise ValueError("adapter and adapter_sha256 must be supplied together")
    if adapter_sha256 and (
        len(adapter_sha256) != 64
        or any(character not in "0123456789abcdef" for character in adapter_sha256)
    ):
        raise ValueError("adapter_sha256 must be a lowercase SHA-256 digest")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            records.append(value)
    return records


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(record, ensure_ascii=False, sort_keys=True)
    fd = os.open(output, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(fd, encoded.encode("utf-8") + b"\n")
        os.fsync(fd)
    finally:
        os.close(fd)


def key(record: dict[str, Any]) -> tuple[str, str, str, str, str, int]:
    return (
        str(record["model"]),
        str(record["method"]),
        str(record["checkpoint"]),
        str(record["benchmark"]),
        str(record["problem_id"]),
        int(record["sample_index"]),
    )


def validate_unique_complete(
    records: Iterable[dict[str, Any]],
    expected_problem_ids: Iterable[str],
    samples_per_problem: int,
) -> None:
    materialized = list(records)
    keys = [key(record) for record in materialized]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate generation keys detected")
    observed = {(str(r["problem_id"]), int(r["sample_index"])) for r in materialized}
    expected = {
        (str(problem_id), sample_index)
        for problem_id in expected_problem_ids
        for sample_index in range(samples_per_problem)
    }
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise ValueError(
            f"incomplete generations: missing={missing[:10]} ({len(missing)} total), "
            f"extra={extra[:10]} ({len(extra)} total)"
        )
