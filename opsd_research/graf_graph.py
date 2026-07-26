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


def graph_builder_prompt(
    problem: str, reference_solution: str, *, graph_budget: int = 24
) -> str:
    """Prompt an offline builder while making the answer-mask requirement explicit."""
    return f"""Construct an answer-masked reasoning graph for the following math problem.

Problem:
{problem}

Verified reference solution (available only to identify concepts and checks):
{reference_solution}

Return JSON with `forks`. Each fork has `fork_id`, `state`, and a variable
number of genuinely distinct `actions`. Use only as many forks/actions as the
problem warrants, while staying within a total graph budget of {graph_budget}
action records. Every retained fork must have at least two alternatives.
Each action has `action_id`, `description`, `status`, `validation_test`, and
`recovery_action`. Status must be one of: viable, conditionally_viable, risky,
invalid, dead_end, recoverable, redundant.

Never include the final answer, any boxed expression, a unique numerical
intermediate from the reference, or a copied reference-solution phrase. Action
descriptions must describe a general mathematical operation, not its result.
Do not reuse even three consecutive words from the reference solution: rewrite
every operation independently and tersely. Prefer short general descriptions
that could apply to a related problem.
Output only the JSON object—no prose, Markdown, or explanation."""


def graph_sanitizer_prompt(problem: str, *, graph_budget: int = 24) -> str:
    """Request a clean graph without exposing either solution or candidate text.

    This is only used after the first candidate has passed answer-leak checks
    but failed an exact reference-fragment check. The fallback gets only the
    problem, so it cannot echo a phrase from the rejected candidate or from the
    reference solution. Its result is still fully revalidated.
    """
    return f"""Construct an answer-masked reasoning graph for the math problem below.

Problem:
{problem}

Construct a fresh answer-masked reasoning graph using only this problem.
Return JSON with only `forks`; each fork has `fork_id`, `state`, and a
variable number of distinct actions within a total budget of {graph_budget}.
Each action has `action_id`, `description`, `status`,
`validation_test`, and `recovery_action`. Status must be one of: viable,
conditionally_viable, risky, invalid, dead_end, recoverable, redundant.
Do not include a final answer, a boxed expression, a numerical intermediate,
or any explanation outside the JSON object."""


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
    payload: dict[str, Any], *, problem: str, reference_solution: str,
    max_forks: int = 6, max_actions_per_fork: int = 6,
    graph_budget: int = 24, check_reference_fragments: bool = True,
) -> ReasoningGraph:
    """Validate a builder payload and reject rather than redact leakage.

    Redaction would leave unknown semantic leakage in a target.  A failed graph
    is rebuilt with a new builder seed and recorded as rejected by the cache.
    """
    if not isinstance(payload.get("forks"), list) or not payload["forks"]:
        raise ValueError("graph requires a nonempty forks list")
    if len(payload["forks"]) > max_forks:
        raise ValueError(f"graph exceeds max_forks={max_forks}")
    if graph_budget < 2:
        raise ValueError("graph_budget must permit at least one decision fork")
    answer = extract_last_boxed(reference_solution)
    fragments = _reference_fragments(reference_solution)
    for field in _all_text(payload):
        normalized = _normalise(field)
        if "\\boxed" in field or "final answer" in normalized:
            raise ValueError("graph contains an answer-format leak")
        if answer and re.search(rf"(?<![A-Za-z0-9]){re.escape(answer)}(?![A-Za-z0-9])", field):
            raise ValueError("graph contains the reference answer")
        if check_reference_fragments and any(
            fragment and fragment in normalized for fragment in fragments
        ):
            raise ValueError("graph copies a five-word reference fragment")
    forks: list[GraphFork] = []
    seen_forks: set[str] = set()
    action_count = 0
    for raw_fork in payload["forks"]:
        fork_id = str(raw_fork.get("fork_id", "")).strip()
        state = str(raw_fork.get("state", "")).strip()
        if not fork_id or not state or fork_id in seen_forks:
            raise ValueError("graph fork IDs must be unique and nonempty")
        seen_forks.add(fork_id)
        raw_actions = raw_fork.get("actions")
        if (
            not isinstance(raw_actions, list)
            or not 2 <= len(raw_actions) <= max_actions_per_fork
        ):
            raise ValueError(
                "each fork requires 2 through max_actions_per_fork distinct actions"
            )
        action_count += len(raw_actions)
        if action_count > graph_budget:
            raise ValueError(f"graph exceeds graph_budget={graph_budget} action records")
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


def graph_critic_prompt(
    problem: str, reference_solution: str, candidate_graph: dict[str, Any], *, graph_budget: int = 24
) -> str:
    """Ask the privileged cache builder to revise, not merely format, a graph.

    The reference is available only in this offline call. The returned graph is
    subsequently subjected to the ordinary answer-masking and reference-fragment
    checks before it can enter the immutable cache.
    """
    return f"""You are auditing a proposed strategy graph for a math problem.
You may use the reference solution only to detect omitted cases, invalid
assumptions, weak checks, and ineffective recovery actions. Return a revised
graph that improves the plan, but never include a final answer, numerical
result, or wording copied from the reference.

Problem:
{problem}

Reference solution (private; do not copy it into JSON):
{reference_solution}

Candidate answer-masked graph:
{json.dumps(candidate_graph, ensure_ascii=False, sort_keys=True)}

Return JSON with only `forks`. Each fork has `fork_id`, `state`, and a variable
number of actions within a total action budget of {graph_budget}. Each action
has `action_id`, `description`, `status`, `validation_test`, and
`recovery_action`. Status is one of viable, conditionally_viable, risky,
invalid, dead_end. Preserve multiple genuinely useful approaches; do not
invent a fixed number of actions or statuses. JSON only."""


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


def render_graph_scaffold(graph: ReasoningGraph) -> str:
    """Render a non-answer graph as a compact privileged teacher scaffold."""
    lines = [
        "Use this answer-masked strategy graph as a scaffold. It contains no final "
        "answer; independently solve and verify the problem."
    ]
    for fork in graph.forks:
        lines.append(f"State {fork.fork_id}: {fork.state}")
        for action in fork.actions:
            lines.append(
                f"- {action.action_id} [{action.status}]: {action.description} "
                f"Check: {action.validation_test} Recovery: {action.recovery_action}"
            )
    return "\n".join(lines)


def write_graph_cache_record(path: Path, graph: ReasoningGraph, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"graph": asdict(graph), "graph_sha256": graph.sha256, "metadata": metadata}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
