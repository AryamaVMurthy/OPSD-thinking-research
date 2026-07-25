"""Answer-masked GRAF graph schema, leakage checks, and branch targets."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .generation_common import extract_last_boxed


ALLOWED_BRANCH_STATUS = {
    "viable",
    "conditionally_viable",
    "risky",
    "invalid",
    "dead_end",
    "recoverable",
    "redundant",
}
_REFERENCE_FRAGMENT = re.compile(r"\S+(?:\s+\S+){3,}")


@dataclass(frozen=True)
class GraphAction:
    action_id: str
    description: str
    status: str
    validation_test: str
    recovery_action: str


@dataclass(frozen=True)
class GraphFork:
    fork_id: str
    state: str
    actions: tuple[GraphAction, ...]


@dataclass(frozen=True)
class ReasoningGraph:
    problem_sha256: str
    forks: tuple[GraphFork, ...]
    schema_version: int = 1

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def graph_builder_prompt(problem: str, reference_solution: str) -> str:
    """Prompt an offline builder while making the answer-mask requirement explicit."""
    return f"""Construct an answer-masked reasoning graph for the following math problem.

Problem:
{problem}

Verified reference solution (available only to identify concepts and checks):
{reference_solution}

Return JSON with `forks`. Each fork has `fork_id`, `state`, and 2-3 `actions`.
Each action has `action_id`, `description`, `status`, `validation_test`, and
`recovery_action`. Status must be one of: viable, conditionally_viable, risky,
invalid, dead_end, recoverable, redundant.

Never include the final answer, any boxed expression, a unique numerical
intermediate from the reference, or a copied reference-solution phrase. Action
descriptions must describe a general mathematical operation, not its result."""


def _normalise(text: str) -> str:
    return " ".join(text.lower().split())


def _reference_fragments(solution: str) -> set[str]:
    """Four-word reference fragments catch direct solution copying cheaply."""
    tokens = _normalise(solution).split()
    return {" ".join(tokens[index : index + 5]) for index in range(max(0, len(tokens) - 4))}


def _all_text(payload: dict[str, Any]) -> list[str]:
    text: list[str] = []
    for fork in payload.get("forks", []):
        text.append(str(fork.get("state", "")))
        for action in fork.get("actions", []):
            for field in ("description", "validation_test", "recovery_action"):
                text.append(str(action.get(field, "")))
    return text


def parse_answer_masked_graph(
    payload: dict[str, Any], *, problem: str, reference_solution: str, max_forks: int = 2
) -> ReasoningGraph:
    """Validate a builder payload and reject rather than redact leakage.

    Redaction would leave unknown semantic leakage in a target.  A failed graph
    is rebuilt with a new builder seed and recorded as rejected by the cache.
    """
    if not isinstance(payload.get("forks"), list) or not payload["forks"]:
        raise ValueError("graph requires a nonempty forks list")
    if len(payload["forks"]) > max_forks:
        raise ValueError(f"graph exceeds max_forks={max_forks}")
    answer = extract_last_boxed(reference_solution)
    fragments = _reference_fragments(reference_solution)
    for field in _all_text(payload):
        normalized = _normalise(field)
        if "\\boxed" in field or "final answer" in normalized:
            raise ValueError("graph contains an answer-format leak")
        if answer and re.search(rf"(?<![A-Za-z0-9]){re.escape(answer)}(?![A-Za-z0-9])", field):
            raise ValueError("graph contains the reference answer")
        if any(fragment and fragment in normalized for fragment in fragments):
            raise ValueError("graph copies a five-word reference fragment")
    forks: list[GraphFork] = []
    seen_forks: set[str] = set()
    for raw_fork in payload["forks"]:
        fork_id = str(raw_fork.get("fork_id", "")).strip()
        state = str(raw_fork.get("state", "")).strip()
        if not fork_id or not state or fork_id in seen_forks:
            raise ValueError("graph fork IDs must be unique and nonempty")
        seen_forks.add(fork_id)
        raw_actions = raw_fork.get("actions")
        if not isinstance(raw_actions, list) or not 2 <= len(raw_actions) <= 3:
            raise ValueError("each fork requires 2-3 canonical actions")
        actions: list[GraphAction] = []
        seen_actions: set[str] = set()
        for raw_action in raw_actions:
            action = GraphAction(
                action_id=str(raw_action.get("action_id", "")).strip(),
                description=str(raw_action.get("description", "")).strip(),
                status=str(raw_action.get("status", "")).strip(),
                validation_test=str(raw_action.get("validation_test", "")).strip(),
                recovery_action=str(raw_action.get("recovery_action", "")).strip(),
            )
            if not all(asdict(action).values()) or action.action_id in seen_actions:
                raise ValueError("graph action fields and IDs must be nonempty")
            if action.status not in ALLOWED_BRANCH_STATUS:
                raise ValueError(f"unsupported action status {action.status!r}")
            seen_actions.add(action.action_id)
            actions.append(action)
        forks.append(GraphFork(fork_id=fork_id, state=state, actions=tuple(actions)))
    return ReasoningGraph(
        problem_sha256=hashlib.sha256(problem.encode("utf-8")).hexdigest(),
        forks=tuple(forks),
    )


def branch_target(
    actions: tuple[GraphAction, ...], viability: dict[str, float], *, temperature: float
) -> list[float]:
    """Convert forced-continuation success estimates to a target distribution."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scores: list[float] = []
    for action in actions:
        if action.status in {"invalid", "dead_end"}:
            scores.append(float("-inf"))
            continue
        value = viability.get(action.action_id)
        if value is None or not 0.0 <= value <= 1.0:
            raise ValueError(f"missing or invalid viability for {action.action_id!r}")
        scores.append(value / temperature)
    finite = [score for score in scores if math.isfinite(score)]
    if not finite:
        raise ValueError("a fork must retain at least one non-invalid action")
    maximum = max(finite)
    weights = [0.0 if not math.isfinite(score) else math.exp(score - maximum) for score in scores]
    normalizer = sum(weights)
    return [weight / normalizer for weight in weights]


def write_graph_cache_record(path: Path, graph: ReasoningGraph, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"graph": asdict(graph), "graph_sha256": graph.sha256, "metadata": metadata}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
