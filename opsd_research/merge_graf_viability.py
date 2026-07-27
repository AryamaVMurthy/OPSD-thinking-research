"""Merge independently generated viability shards into one immutable cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .graf_cache import validate_graph_cache_manifest
from .graf_actions import ASSISTANT_ACTION_PREFIX_PROTOCOL, RECOVERY_ACTION_PREFIX_PROTOCOL
from .records import append_jsonl, read_jsonl


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-manifest", required=True, type=Path)
    parser.add_argument("--shard", required=True, type=Path, action="append")
    parser.add_argument("--trial-shard", type=Path, action="append")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--trials-output", type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--samples-per-action", required=True, type=int)
    parser.add_argument("--temperature", required=True, type=float)
    parser.add_argument("--max-completion-tokens", required=True, type=int)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--forced-prefix-protocol", default=ASSISTANT_ACTION_PREFIX_PROTOCOL)
    return parser.parse_args()


def main() -> None:
    args = _args()
    if bool(args.trial_shard) != bool(args.trials_output):
        raise SystemExit("--trial-shard and --trials-output must be supplied together")
    immutable_outputs = [args.output, args.manifest]
    if args.trials_output is not None:
        immutable_outputs.append(args.trials_output)
    if any(path.exists() for path in immutable_outputs):
        raise SystemExit("refusing to overwrite an immutable viability cache or manifest")
    if args.forced_prefix_protocol not in {
        ASSISTANT_ACTION_PREFIX_PROTOCOL, RECOVERY_ACTION_PREFIX_PROTOCOL,
    }:
        raise SystemExit("unsupported forced action-prefix protocol")
    if args.max_completion_tokens < 1:
        raise SystemExit("max completion tokens must be positive")
    graph_manifest = validate_graph_cache_manifest(args.graph_manifest)
    graph_path = Path(str(graph_manifest["cache"]))
    if not graph_path.is_absolute():
        graph_path = args.graph_manifest.parent / graph_path
    expected = {
        int(record["example_index"]): str(record["graph_sha256"])
        for record in read_jsonl(graph_path)
        if record.get("accepted")
    }
    records: dict[int, dict] = {}
    for shard in args.shard:
        if not shard.is_file():
            raise SystemExit(f"viability shard does not exist: {shard}")
        for record in read_jsonl(shard):
            index = int(record.get("example_index", -1))
            if index in records:
                raise SystemExit(f"duplicate viability target for example {index}")
            if expected.get(index) != record.get("graph_sha256"):
                raise SystemExit(f"viability target {index} is not joined to the graph cache")
            if int(record.get("samples_per_action", -1)) != args.samples_per_action:
                raise SystemExit("viability shard samples-per-action mismatch")
            if float(record.get("temperature", -1)) != args.temperature:
                raise SystemExit("viability shard temperature mismatch")
            if int(record.get("max_completion_tokens", -1)) != args.max_completion_tokens:
                raise SystemExit("viability shard max-completion-tokens mismatch")
            if record.get("forced_prefix_protocol") != args.forced_prefix_protocol:
                raise SystemExit("viability shard action-prefix protocol mismatch")
            if record.get("model") != args.model or record.get("model_revision") != args.model_revision:
                raise SystemExit("viability shard model pin mismatch")
            records[index] = record
    if not records:
        raise SystemExit("no viability records to merge")

    trials: dict[str, dict] = {}
    if args.trial_shard:
        expected_trial_links: dict[str, tuple[int, str, str]] = {}
        for index, record in records.items():
            for fork in record.get("fork_targets", []):
                fork_id = str(fork.get("fork_id", ""))
                links = fork.get("trial_ids_by_action")
                if not isinstance(links, dict):
                    raise SystemExit(
                        f"viability target {index}/{fork_id} lacks raw trial links"
                    )
                for action_id, trial_ids in links.items():
                    if not isinstance(trial_ids, list) or not trial_ids:
                        raise SystemExit(
                            f"viability target {index}/{fork_id}/{action_id} "
                            "has no raw trial links"
                        )
                    for trial_id in trial_ids:
                        identity = (index, fork_id, str(action_id))
                        if not isinstance(trial_id, str) or not trial_id:
                            raise SystemExit("viability trial IDs must be nonempty strings")
                        if trial_id in expected_trial_links:
                            raise SystemExit(f"duplicate viability trial link: {trial_id}")
                        expected_trial_links[trial_id] = identity

        for shard in args.trial_shard:
            if not shard.is_file():
                raise SystemExit(f"viability trial shard does not exist: {shard}")
            for trial in read_jsonl(shard):
                if int(trial.get("schema_version", -1)) != 1:
                    raise SystemExit("unsupported viability trial schema")
                trial_id = trial.get("trial_id")
                if not isinstance(trial_id, str) or not trial_id:
                    raise SystemExit("viability trial ID must be a nonempty string")
                if trial_id in trials:
                    raise SystemExit(f"duplicate viability trial record: {trial_id}")
                identity = (
                    int(trial.get("example_index", -1)),
                    str(trial.get("fork_id", "")),
                    str(trial.get("action_id", "")),
                )
                if expected_trial_links.get(trial_id) != identity:
                    raise SystemExit(
                        f"viability trial {trial_id} is not joined to its aggregate target"
                    )
                if int(trial.get("sample_index", -1)) < 0:
                    raise SystemExit(
                        f"viability trial {trial_id} has an invalid sample index"
                    )
                trials[trial_id] = trial
        missing = sorted(set(expected_trial_links) - set(trials))
        if missing:
            raise SystemExit(
                f"viability aggregate references missing raw trials: {missing[:5]}"
            )

    for index in sorted(records):
        append_jsonl(args.output, records[index])
    if args.trials_output is not None:
        for trial in sorted(
            trials.values(),
            key=lambda trial: (
                int(trial["example_index"]),
                str(trial["fork_id"]),
                str(trial["action_id"]),
                int(trial["sample_index"]),
                str(trial["trial_id"]),
            ),
        ):
            append_jsonl(args.trials_output, trial)
    manifest = {
        "schema_version": 1,
        "graph_cache_sha256": graph_manifest["cache_sha256"],
        "viability_cache": str(args.output),
        "viability_cache_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "examples": len(records),
        "samples_per_action": args.samples_per_action,
        "temperature": args.temperature,
        "max_completion_tokens": args.max_completion_tokens,
        "forced_prefix_protocol": args.forced_prefix_protocol,
        "model": args.model,
        "model_revision": args.model_revision,
        "seed": args.seed,
        "shards": [str(path) for path in args.shard],
        "trial_evidence_available": args.trials_output is not None,
    }
    if args.trials_output is not None:
        manifest.update(
            {
                "trial_cache": str(args.trials_output),
                "trial_cache_sha256": hashlib.sha256(
                    args.trials_output.read_bytes()
                ).hexdigest(),
                "trial_records": len(trials),
                "trial_schema_version": 1,
                "trial_shards": [str(path) for path in args.trial_shard],
            }
        )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
