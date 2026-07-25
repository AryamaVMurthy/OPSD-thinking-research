"""Estimate forced student-context branch viability for a frozen GRAF cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .graf_actions import ASSISTANT_ACTION_PREFIX_PROTOCOL, action_continuation
from .generation_common import extract_last_boxed, grade_math
from .graf_cache import validate_graph_cache_manifest
from .graf_graph import GraphAction, branch_target
from .prompts import math_messages, render_thinking_prompt
from .records import append_jsonl, read_jsonl


DEFAULT_MODEL = "Qwen/Qwen3-4B"
DEFAULT_MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
TRAINING_DATASET_REVISION = "1435fb21d4fecc8ad4966a26f22a874cf2b527f1"


def forced_action_prompt(tokenizer: Any, problem: str, action: str) -> str:
    """Force the *same assistant continuation* scored by GRAF's branch loss.

    The ordinary student message remains untouched.  The graph action is
    appended after Qwen's thinking-enabled assistant prefix, so empirical
    viability answers the relevant question: can the student finish after it
    has emitted this exact action, rather than after the user merely suggested
    it?
    """
    return render_thinking_prompt(tokenizer, math_messages(problem)) + action_continuation(action)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build forced-continuation viability targets")
    parser.add_argument("--graph-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--start-index", type=int, default=0,
        help="Offset within accepted cache rows; used only by independently generated shards.",
    )
    parser.add_argument("--limit", required=True, type=int)
    parser.add_argument("--samples-per-action", required=True, type=int)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if (
        args.limit < 1
        or args.start_index < 0
        or args.samples_per_action < 1
        or args.temperature <= 0
    ):
        raise SystemExit("limit, samples-per-action, and temperature must be positive")
    if args.model != DEFAULT_MODEL or args.model_revision != DEFAULT_MODEL_REVISION:
        raise SystemExit("GRAF viability building is pinned to Qwen3-4B@1cfa9a7")
    if args.output.exists() or (args.manifest is not None and args.manifest.exists()):
        raise SystemExit("refusing to alter an existing viability cache or manifest")
    graph_manifest = validate_graph_cache_manifest(args.graph_manifest)
    graph_cache_path = Path(str(graph_manifest["cache"]))
    if not graph_cache_path.is_absolute():
        graph_cache_path = args.graph_manifest.parent / graph_cache_path
    accepted = [record for record in read_jsonl(graph_cache_path) if record.get("accepted")]
    accepted = accepted[args.start_index : args.start_index + args.limit]
    if not accepted:
        raise SystemExit("graph cache contains no accepted records")

    from .training_data import load_math_cot_20k
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = load_math_cot_20k()["train"]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.model_revision, trust_remote_code=True
    )
    requests: list[tuple[int, str, str, str, str]] = []
    prompts: list[str] = []
    for record in accepted:
        row = rows[int(record["example_index"])]
        graph = record["graph"]
        for fork in graph["forks"]:
            for action in fork["actions"]:
                if action["status"] in {"invalid", "dead_end"}:
                    continue
                for sample_index in range(args.samples_per_action):
                    requests.append((
                        int(record["example_index"]), str(fork["fork_id"]),
                        str(action["action_id"]), str(row["response"]), str(action["description"]),
                    ))
                    prompts.append(forced_action_prompt(
                        tokenizer, str(row["question"]), str(action["description"])
                    ))
    llm = LLM(
        model=args.model, revision=args.model_revision, dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size, max_model_len=40960,
        gpu_memory_utilization=0.90, enforce_eager=True, trust_remote_code=True,
    )
    outputs = llm.generate(
        prompts,
        SamplingParams(
            temperature=args.temperature, top_p=0.95, max_tokens=38912, seed=args.seed
        ),
    )
    successes: dict[tuple[int, str, str], list[bool]] = defaultdict(list)
    for request, output in zip(requests, outputs, strict=True):
        index, fork_id, action_id, answer, _ = request
        successes[(index, fork_id, action_id)].append(
            grade_math(extract_last_boxed(output.outputs[0].text), answer)
        )
    for record in accepted:
        fork_targets = []
        for raw_fork in record["graph"]["forks"]:
            actions = tuple(GraphAction(**action) for action in raw_fork["actions"])
            viability = {
                action.action_id: sum(successes[(int(record["example_index"]), raw_fork["fork_id"], action.action_id)])
                / args.samples_per_action
                for action in actions
                if action.status not in {"invalid", "dead_end"}
            }
            fork_targets.append({
                "fork_id": raw_fork["fork_id"],
                "action_ids": [action.action_id for action in actions],
                "viability": viability,
                "target": branch_target(actions, viability, temperature=args.temperature),
            })
        append_jsonl(args.output, {
            "schema_version": 1,
            "example_index": record["example_index"],
            "graph_sha256": record["graph_sha256"],
            "fork_targets": fork_targets,
            "samples_per_action": args.samples_per_action,
            "temperature": args.temperature,
            "forced_prefix_protocol": ASSISTANT_ACTION_PREFIX_PROTOCOL,
            "model": args.model,
            "model_revision": args.model_revision,
        })
    manifest = {
        "schema_version": 1,
        "graph_cache_sha256": graph_manifest["cache_sha256"],
        "viability_cache": str(args.output),
        "viability_cache_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "examples": len(accepted),
        "samples_per_action": args.samples_per_action,
        "temperature": args.temperature,
        "forced_prefix_protocol": ASSISTANT_ACTION_PREFIX_PROTOCOL,
        "model": args.model,
        "model_revision": args.model_revision,
        "training_dataset_revision": TRAINING_DATASET_REVISION,
        "seed": args.seed,
    }
    if args.manifest is not None:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
