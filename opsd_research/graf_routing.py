"""Join immutable GRAF graphs with forced-continuation branch targets."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

from .graf_actions import ASSISTANT_ACTION_PREFIX_PROTOCOL
from .graf_cache import validate_graph_cache_manifest


@dataclass(frozen=True)
class ActionTarget:
    action_id: str
    description: str
    target_probability: float


@dataclass(frozen=True)
class ForkTarget:
    fork_id: str
    actions: tuple[ActionTarget, ...]
    information_weight: float = 1.0


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _resolve_manifest_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest_path.parent / path


def load_routing_targets(
    graph_manifest_path: str | Path,
    viability_manifest_path: str | Path,
    *,
    min_target_margin: float = 0.0,
    min_target_information: float = 0.0,
    target_information_quantile: float = 0.0,
    information_weighting: bool = False,
) -> dict[int, tuple[ForkTarget, ...]]:
    """Verify caches and optionally retain outcome-differential branch targets.

    A uniform empirical target supplies no preference between actions.  Such a
    fork should not contribute a KL/entropy action gradient, but its example
    still remains in the base OPSD data stream.  Keeping an empty tuple for
    that identity preserves this separation and makes the threshold auditable.
    """
    if not 0.0 <= min_target_margin <= 1.0:
        raise ValueError("min_target_margin must be in [0, 1]")
    if not 0.0 <= min_target_information <= 1.0:
        raise ValueError("min_target_information must be in [0, 1]")
    if not 0.0 <= target_information_quantile <= 1.0:
        raise ValueError("target_information_quantile must be in [0, 1]")
    if min_target_information and target_information_quantile:
        raise ValueError("choose either min_target_information or target_information_quantile")
    graph_manifest_path = Path(graph_manifest_path)
    viability_manifest_path = Path(viability_manifest_path)
    graph_manifest = validate_graph_cache_manifest(graph_manifest_path)
    try:
        viability_manifest = json.loads(viability_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid viability manifest: {error}") from error
    if viability_manifest.get("schema_version") != 1:
        raise ValueError("unsupported viability manifest schema")
    if viability_manifest.get("forced_prefix_protocol") != ASSISTANT_ACTION_PREFIX_PROTOCOL:
        raise ValueError("viability cache uses an incompatible action-prefix protocol")
    if viability_manifest.get("graph_cache_sha256") != graph_manifest["cache_sha256"]:
        raise ValueError("viability cache was built from a different graph cache")
    viability_path = _resolve_manifest_path(
        viability_manifest_path, str(viability_manifest.get("viability_cache", ""))
    )
    if not viability_path.is_file():
        raise ValueError(f"viability cache does not exist: {viability_path}")
    observed = hashlib.sha256(viability_path.read_bytes()).hexdigest()
    if observed != viability_manifest.get("viability_cache_sha256"):
        raise ValueError("viability cache SHA-256 does not match its manifest")

    graph_path = _resolve_manifest_path(graph_manifest_path, str(graph_manifest["cache"]))
    graphs = {
        int(record["example_index"]): record
        for record in _read_jsonl(graph_path)
        if record.get("accepted")
    }
    records = _read_jsonl(viability_path)

    def target_information(values: list[float]) -> float:
        uniform = 1.0 / len(values)
        return sum(value * math.log(value / uniform) for value in values if value > 0.0)

    # When configured, the threshold is derived from the measured cache rather
    # than from a particular count of viable/invalid actions.  It works for
    # every arity and retains the upper (1-q) portion of non-uniform target
    # distributions.  q=0 keeps the explicit threshold/no-filter behavior.
    adaptive_information = min_target_information
    if target_information_quantile:
        observed: list[float] = []
        for record in records:
            for fork in record.get("fork_targets", []):
                values = [float(value) for value in fork.get("target", [])]
                if values and all(value >= 0.0 for value in values) and abs(sum(values) - 1.0) <= 1e-6:
                    information = target_information(values)
                    if information > 0.0:
                        observed.append(information)
        if not observed:
            raise ValueError("information-quantile routing found no non-uniform targets")
        observed.sort()
        cutoff_index = min(
            len(observed) - 1, int(math.floor(target_information_quantile * len(observed)))
        )
        adaptive_information = observed[cutoff_index]

    targets: dict[int, tuple[ForkTarget, ...]] = {}
    for record in records:
        if record.get("forced_prefix_protocol") != ASSISTANT_ACTION_PREFIX_PROTOCOL:
            raise ValueError("viability record uses an incompatible action-prefix protocol")
        index = int(record["example_index"])
        graph = graphs.get(index)
        if graph is None or record.get("graph_sha256") != graph.get("graph_sha256"):
            raise ValueError(f"viability record {index} does not match an accepted graph")
        descriptions = {
            str(action["action_id"]): str(action["description"])
            for fork in graph["graph"]["forks"]
            for action in fork["actions"]
        }
        forks = []
        for fork in record.get("fork_targets", []):
            action_ids = [str(action_id) for action_id in fork["action_ids"]]
            values = [float(value) for value in fork["target"]]
            if len(action_ids) != len(values) or not action_ids:
                raise ValueError(f"invalid target shape for example {index}")
            if any(value < 0 for value in values) or abs(sum(values) - 1.0) > 1e-6:
                raise ValueError(f"invalid target probability for example {index}")
            if any(action_id not in descriptions for action_id in action_ids):
                raise ValueError(f"unknown action ID for example {index}")
            sorted_values = sorted(values, reverse=True)
            margin = sorted_values[0] - sorted_values[1] if len(sorted_values) > 1 else 0.0
            if margin < min_target_margin:
                continue
            # Retain any target that carries information about its action set.
            # This differs from a top-two winner margin: [viable, viable,
            # invalid] should preserve the two viable alternatives while
            # explicitly suppressing the invalid one.
            information = target_information(values)
            if information < adaptive_information:
                continue
            if information_weighting and information == 0.0:
                continue
            # Normalizing by the maximum KL for this action arity gives a
            # continuous [0, 1] reliability weight without assuming how many
            # actions are viable or invalid.
            information_weight = (
                information / math.log(len(values)) if information_weighting else 1.0
            )
            forks.append(ForkTarget(
                fork_id=str(fork["fork_id"]),
                actions=tuple(
                    ActionTarget(action_id, descriptions[action_id], value)
                    for action_id, value in zip(action_ids, values, strict=True)
                ),
                information_weight=information_weight,
            ))
        targets[index] = tuple(forks)
    if len(targets) != int(viability_manifest.get("examples", -1)):
        raise ValueError("viability manifest count does not match target records")
    return targets
