"""Immutable GRAF-OPSD candidate ledger and promotion gates.

This module deliberately contains no model code.  It prevents an autoresearch
loop from silently mutating protocol, benchmark, or promotion criteria while
it is looking at development results.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    parent_id: str | None
    stage: str
    hypothesis: str
    config: dict[str, Any]
    changed_fields: tuple[str, ...]

    def canonical_payload(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_payload().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DevelopmentResult:
    candidate_id: str
    avg_at_12_delta: float
    delta_ci_lower: float
    delta_ci_upper: float
    samples_per_problem: int
    benchmark: str


def validate_candidate(candidate: Candidate, allowed_mutations: set[str]) -> None:
    if not candidate.candidate_id or any(char.isspace() for char in candidate.candidate_id):
        raise ValueError("candidate_id must be nonempty and contain no whitespace")
    if not candidate.hypothesis.strip():
        raise ValueError("candidate requires a hypothesis")
    changed = set(candidate.changed_fields)
    if not changed:
        raise ValueError("candidate must declare exactly one experimental change")
    if not changed.issubset(allowed_mutations):
        raise ValueError(f"candidate has forbidden mutations: {sorted(changed - allowed_mutations)}")
    if len(changed) != 1:
        raise ValueError("candidate must change exactly one allowed field")


def qualifies_for_promotion(
    result: DevelopmentResult,
    *,
    minimum_delta: float,
    development_benchmark: str = "aime24",
) -> bool:
    return (
        result.benchmark == development_benchmark
        and result.samples_per_problem == 12
        and result.avg_at_12_delta >= minimum_delta
        and result.delta_ci_lower > 0.0
        and result.delta_ci_upper > 0.0
    )


def append_ledger(path: Path, event: dict[str, Any]) -> str:
    """Append a canonical JSONL event and return its SHA-256 digest."""
    payload = json.dumps(event, sort_keys=True, separators=(",", ":"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload + "\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
