"""Natural teacher evidence joining blind audits and measured continuations."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .adaptive_viability import beta_credible_interval
from .ch_dossier import accepted_teacher_dossiers
from .graf_cache import validate_graph_cache_manifest
from .graf_routing import load_routing_targets


def _resolve(manifest_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else manifest_path.parent / path


def _jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid JSONL at {path}:{line_number}"
            ) from error
        if not isinstance(record, dict):
            raise ValueError(f"non-object JSONL at {path}:{line_number}")
        records.append(record)
    return records


def _render_fork(
    graph_fork: dict[str, Any],
    viability_fork: dict[str, Any],
    *,
    default_trials: int,
    credible_level: float,
) -> str:
    graph_actions = {
        str(action["action_id"]): action
        for action in graph_fork.get("actions", [])
    }
    action_ids = [
        str(action_id) for action_id in viability_fork.get("action_ids", [])
    ]
    if not action_ids or any(action_id not in graph_actions for action_id in action_ids):
        raise ValueError("viability fork refers to unknown graph actions")
    raw_viability = viability_fork.get("viability")
    if not isinstance(raw_viability, dict):
        raise ValueError("fluid teacher context requires empirical viability")
    samples_by_action = viability_fork.get("samples_by_action", {})
    if not isinstance(samples_by_action, dict):
        raise ValueError("samples_by_action must be a mapping")
    lines = [
        f"At the reasoning state “{str(graph_fork.get('state', '')).strip()}”:"
    ]
    for action_id in action_ids:
        action = graph_actions[action_id]
        description = str(action["description"]).strip()
        status = str(action.get("status", "")).strip()
        if action_id not in raw_viability:
            lines.append(
                f"- “{description}” was marked {status or 'invalid'} before "
                "continuation sampling."
            )
            continue
        try:
            trials = int(samples_by_action.get(action_id, default_trials))
            rate = float(raw_viability[action_id])
        except (TypeError, ValueError) as error:
            raise ValueError("invalid action viability evidence") from error
        if trials < 1 or not 0.0 <= rate <= 1.0:
            raise ValueError("invalid action viability evidence")
        successes_float = rate * trials
        successes = int(round(successes_float))
        if not math.isclose(
            successes_float, successes, rel_tol=0.0, abs_tol=1e-6
        ):
            raise ValueError(
                "viability rate is inconsistent with its action sample count"
            )
        lower, upper = beta_credible_interval(
            successes,
            trials,
            credible_level=credible_level,
        )
        lines.append(
            f"- Continuing with “{description}” reached the verified answer "
            f"in {successes} of {trials} frozen-policy completions. Its "
            f"{100 * credible_level:.0f}% Beta interval is approximately "
            f"{lower:.2f}–{upper:.2f}; treat this as small-sample evidence, "
            "not certainty."
        )
    return "\n".join(lines)


def fluid_teacher_contexts(
    dossier_manifest_path: str | Path,
    graph_manifest_path: str | Path,
    viability_manifest_path: str | Path,
    *,
    credible_level: float = 0.9,
) -> dict[int, str]:
    """Return every dossier, enriched where verified routing evidence exists."""
    if not 0.0 < credible_level < 1.0:
        raise ValueError("credible_level must be in (0, 1)")
    dossier_manifest_path = Path(dossier_manifest_path)
    graph_manifest_path = Path(graph_manifest_path)
    viability_manifest_path = Path(viability_manifest_path)
    contexts = accepted_teacher_dossiers(dossier_manifest_path)

    # Reuse the strict immutable graph/viability join before rendering any
    # teacher-visible prose. No target filtering is needed for this validation.
    load_routing_targets(
        graph_manifest_path,
        viability_manifest_path,
    )
    graph_manifest = validate_graph_cache_manifest(graph_manifest_path)
    graph_path = _resolve(
        graph_manifest_path, str(graph_manifest["cache"])
    )
    graphs = {
        int(record["example_index"]): record
        for record in _jsonl(graph_path)
        if record.get("accepted")
    }
    viability_manifest = json.loads(
        viability_manifest_path.read_text(encoding="utf-8")
    )
    viability_path = _resolve(
        viability_manifest_path,
        str(viability_manifest["viability_cache"]),
    )
    enriched = dict(contexts)
    for record in _jsonl(viability_path):
        index = int(record["example_index"])
        if index not in contexts:
            continue
        graph_record = graphs.get(index)
        if (
            graph_record is None
            or record.get("graph_sha256")
            != graph_record.get("graph_sha256")
        ):
            raise ValueError(
                f"fluid teacher evidence does not match graph {index}"
            )
        graph_forks = {
            str(fork["fork_id"]): fork
            for fork in graph_record["graph"].get("forks", [])
        }
        try:
            default_trials = int(record["samples_per_action"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "fluid teacher context requires samples_per_action"
            ) from error
        rendered_forks = []
        for viability_fork in record.get("fork_targets", []):
            fork_id = str(viability_fork["fork_id"])
            if fork_id not in graph_forks:
                raise ValueError(
                    f"unknown viability fork {fork_id!r} for example {index}"
                )
            rendered_forks.append(
                _render_fork(
                    graph_forks[fork_id],
                    viability_fork,
                    default_trials=default_trials,
                    credible_level=credible_level,
                )
            )
        if rendered_forks:
            enriched[index] = "\n\n".join(
                [
                    contexts[index],
                    "Additional empirical continuation evidence for the "
                    "teacher only. Use it flexibly alongside the verified "
                    "solution and comparative audit:",
                    *rendered_forks,
                ]
            )
    return enriched
