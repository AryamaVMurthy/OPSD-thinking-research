"""Estimate forced student-context branch viability for a frozen GRAF cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .graf_actions import (
    ASSISTANT_ACTION_PREFIX_PROTOCOL,
    RECOVERY_ACTION_PREFIX_PROTOCOL,
    action_continuation,
    recovery_conditioned_description,
)
from .adaptive_viability import (
    ActionEvidence,
    beta_credible_interval,
    next_uncertain_action,
)
from .generation_common import extract_last_boxed, grade_math
from .graf_cache import validate_graph_cache_manifest
from .graf_graph import GraphAction, branch_target
from .prompts import math_messages, render_thinking_prompt
from .records import append_jsonl, read_jsonl


DEFAULT_MODEL = "Qwen/Qwen3-4B"
DEFAULT_MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
TRAINING_DATASET_REVISION = "1435fb21d4fecc8ad4966a26f22a874cf2b527f1"


def reference_answer_from_solution(reference_solution: str) -> str:
    """Extract the canonical final answer used as the viability grader target."""
    answer = extract_last_boxed(reference_solution)
    if answer is None or not answer.strip():
        raise ValueError("reference solution lacks a nonempty final boxed answer")
    return answer


def viability_trial_seed(
    base_seed: int,
    example_index: int,
    fork_id: str,
    action_id: str,
    sample_index: int,
) -> int:
    """Derive a trial seed that is stable under sharding and request ordering."""
    if base_seed < 0 or example_index < 0 or sample_index < 0:
        raise ValueError("viability seed inputs must be nonnegative")
    if not fork_id or not action_id:
        raise ValueError("viability seed requires fork and action identities")
    payload = json.dumps(
        {
            "action_id": action_id,
            "base_seed": base_seed,
            "example_index": example_index,
            "fork_id": fork_id,
            "sample_index": sample_index,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


def viability_trial_record(
    *,
    example_index: int,
    fork_id: str,
    action_id: str,
    sample_index: int,
    seed: int,
    prompt: str,
    reference_answer: str,
    completion: str,
    output_token_ids: list[int],
    finish_reason: str,
    legacy_correct: bool,
    model: str,
    model_revision: str,
    forced_prefix_protocol: str,
) -> dict[str, Any]:
    """Return the immutable evidence needed to independently regrade a trial."""
    if example_index < 0 or sample_index < 0 or seed < 0:
        raise ValueError("viability trial indices and seed must be nonnegative")
    strings = {
        "fork_id": fork_id,
        "action_id": action_id,
        "prompt": prompt,
        "reference_answer": reference_answer,
        "completion": completion,
        "finish_reason": finish_reason,
        "model": model,
        "model_revision": model_revision,
        "forced_prefix_protocol": forced_prefix_protocol,
    }
    if any(not str(value).strip() for value in strings.values()):
        raise ValueError("viability trial text and provenance must be nonempty")
    if any(
        not isinstance(token_id, int) or isinstance(token_id, bool)
        for token_id in output_token_ids
    ):
        raise ValueError("output token IDs must be integers")

    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    completion_sha256 = hashlib.sha256(completion.encode("utf-8")).hexdigest()
    token_payload = json.dumps(
        output_token_ids, separators=(",", ":")
    ).encode("utf-8")
    token_sha256 = hashlib.sha256(token_payload).hexdigest()
    identity_payload = json.dumps(
        {
            "action_id": action_id,
            "example_index": example_index,
            "fork_id": fork_id,
            "model_revision": model_revision,
            "sample_index": sample_index,
            "seed": seed,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema_version": 1,
        "trial_id": hashlib.sha256(identity_payload).hexdigest(),
        "example_index": example_index,
        "fork_id": fork_id,
        "action_id": action_id,
        "sample_index": sample_index,
        "generation_seed": seed,
        "prompt_sha256": prompt_sha256,
        "reference_answer": reference_answer,
        "raw_completion": completion,
        "completion_sha256": completion_sha256,
        "output_tokens": len(output_token_ids),
        "output_token_ids_sha256": token_sha256,
        "finish_reason": finish_reason,
        "extracted_answer": extract_last_boxed(completion),
        "legacy_grader": "generation_common.grade_math@v1",
        "legacy_correct": bool(legacy_correct),
        "model": model,
        "model_revision": model_revision,
        "forced_prefix_protocol": forced_prefix_protocol,
    }


@dataclass(frozen=True)
class ViabilityRequest:
    example_index: int
    fork_id: str
    action_id: str
    reference_answer: str
    action_description: str
    sample_index: int
    seed: int
    prompt: str


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
    parser.add_argument(
        "--trials-output",
        type=Path,
        help=(
            "Immutable raw trial JSONL. Defaults beside --output to "
            "<output-stem>.trials.jsonl."
        ),
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--start-index", type=int, default=0,
        help="Offset within accepted cache rows; used only by independently generated shards.",
    )
    parser.add_argument("--limit", required=True, type=int)
    parser.add_argument("--samples-per-action", required=True, type=int)
    parser.add_argument(
        "--adaptive-max-samples-per-action",
        type=int,
        help=(
            "When set above samples-per-action, allocate additional samples "
            "only to actions whose Beta posterior intervals still overlap."
        ),
    )
    parser.add_argument("--credible-level", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-completion-tokens", type=int, default=4096)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recovery-conditioned", action="store_true")
    return parser.parse_args()


def _require_routable_forks(records: list[dict[str, Any]]) -> None:
    """Fail before model loading if an immutable cache cannot define targets."""
    for record in records:
        for fork in record["graph"]["forks"]:
            statuses = [str(action["status"]) for action in fork["actions"]]
            if not any(status not in {"invalid", "dead_end"} for status in statuses):
                raise SystemExit(
                    "graph cache contains a non-routable fork: "
                    f"example_index={record['example_index']} fork_id={fork['fork_id']}"
                )


def main() -> None:
    args = _parse_args()
    trials_output = args.trials_output or args.output.with_name(
        f"{args.output.stem}.trials.jsonl"
    )
    if (
        args.limit < 1
        or args.start_index < 0
        or args.samples_per_action < 1
        or args.temperature <= 0
        or args.max_completion_tokens < 1
        or not 0.0 < args.credible_level < 1.0
    ):
        raise SystemExit("limit, samples-per-action, temperature, and max completion tokens must be positive")
    adaptive_max = args.adaptive_max_samples_per_action
    if adaptive_max is not None and adaptive_max < args.samples_per_action:
        raise SystemExit(
            "adaptive max samples must be at least samples-per-action"
        )
    if args.model != DEFAULT_MODEL or args.model_revision != DEFAULT_MODEL_REVISION:
        raise SystemExit("GRAF viability building is pinned to Qwen3-4B@1cfa9a7")
    if (
        args.output.exists()
        or trials_output.exists()
        or (args.manifest is not None and args.manifest.exists())
    ):
        raise SystemExit(
            "refusing to alter an existing viability cache, trial cache, or manifest"
        )
    action_protocol = (
        RECOVERY_ACTION_PREFIX_PROTOCOL if args.recovery_conditioned
        else ASSISTANT_ACTION_PREFIX_PROTOCOL
    )
    graph_manifest = validate_graph_cache_manifest(args.graph_manifest)
    graph_cache_path = Path(str(graph_manifest["cache"]))
    if not graph_cache_path.is_absolute():
        graph_cache_path = args.graph_manifest.parent / graph_cache_path
    accepted = [record for record in read_jsonl(graph_cache_path) if record.get("accepted")]
    accepted = accepted[args.start_index : args.start_index + args.limit]
    if not accepted:
        raise SystemExit("graph cache contains no accepted records")
    _require_routable_forks(accepted)

    from .training_data import load_math_cot_20k
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = load_math_cot_20k()["train"]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.model_revision, trust_remote_code=True
    )
    requests: list[ViabilityRequest] = []
    action_requests: dict[
        tuple[int, str, str], tuple[str, str, str]
    ] = {}
    for record in accepted:
        row = rows[int(record["example_index"])]
        reference_answer = reference_answer_from_solution(str(row["response"]))
        graph = record["graph"]
        for fork in graph["forks"]:
            for action in fork["actions"]:
                if action["status"] in {"invalid", "dead_end"}:
                    continue
                description = str(action["description"])
                if args.recovery_conditioned:
                    description = recovery_conditioned_description(
                        description,
                        str(action["validation_test"]),
                        str(action["recovery_action"]),
                    )
                key = (
                    int(record["example_index"]),
                    str(fork["fork_id"]),
                    str(action["action_id"]),
                )
                prompt = forced_action_prompt(
                    tokenizer, str(row["question"]), description
                )
                action_requests[key] = (
                    reference_answer,
                    description,
                    prompt,
                )
                for sample_index in range(args.samples_per_action):
                    requests.append(
                        ViabilityRequest(
                            example_index=key[0],
                            fork_id=key[1],
                            action_id=key[2],
                            reference_answer=reference_answer,
                            action_description=description,
                            sample_index=sample_index,
                            seed=viability_trial_seed(
                                args.seed, *key, sample_index
                            ),
                            prompt=prompt,
                        )
                    )
    llm = LLM(
        model=args.model, revision=args.model_revision, dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size, max_model_len=40960,
        gpu_memory_utilization=0.90, enforce_eager=True, trust_remote_code=True,
    )
    def sampling_params(seed: int):
        return SamplingParams(
            temperature=args.temperature, top_p=0.95,
            max_tokens=args.max_completion_tokens, seed=seed
        )

    outputs = llm.generate(
        [request.prompt for request in requests],
        [sampling_params(request.seed) for request in requests],
    )
    successes: dict[tuple[int, str, str], list[bool]] = defaultdict(list)
    trial_ids: dict[tuple[int, str, str], list[str]] = defaultdict(list)

    def record_outputs(
        batch_requests: list[ViabilityRequest], batch_outputs: list[Any]
    ) -> None:
        for request, output in zip(batch_requests, batch_outputs, strict=True):
            generated = output.outputs[0]
            completion = str(generated.text)
            correct = grade_math(
                extract_last_boxed(completion), request.reference_answer
            )
            key = (
                request.example_index,
                request.fork_id,
                request.action_id,
            )
            evidence = viability_trial_record(
                example_index=request.example_index,
                fork_id=request.fork_id,
                action_id=request.action_id,
                sample_index=request.sample_index,
                seed=request.seed,
                prompt=request.prompt,
                reference_answer=request.reference_answer,
                completion=completion,
                output_token_ids=list(generated.token_ids),
                finish_reason=str(
                    getattr(generated, "finish_reason", None) or "unknown"
                ),
                legacy_correct=correct,
                model=args.model,
                model_revision=args.model_revision,
                forced_prefix_protocol=action_protocol,
            )
            append_jsonl(trials_output, evidence)
            successes[key].append(correct)
            trial_ids[key].append(str(evidence["trial_id"]))

    record_outputs(requests, outputs)
    total_completions = len(requests)
    if adaptive_max is not None and adaptive_max > args.samples_per_action:
        fork_actions: dict[
            tuple[int, str], list[tuple[int, str, str]]
        ] = defaultdict(list)
        for key in action_requests:
            fork_actions[key[:2]].append(key)
        for round_index in range(
            adaptive_max - args.samples_per_action
        ):
            selected_keys: list[tuple[int, str, str]] = []
            for keys in fork_actions.values():
                selected = next_uncertain_action(
                    [
                        ActionEvidence(
                            action_id=key[2],
                            successes=sum(successes[key]),
                            trials=len(successes[key]),
                        )
                        for key in keys
                    ],
                    max_trials=adaptive_max,
                    credible_level=args.credible_level,
                )
                if selected is not None:
                    selected_keys.append(
                        next(key for key in keys if key[2] == selected)
                    )
            if not selected_keys:
                break
            adaptive_requests = []
            for key in selected_keys:
                answer, description, prompt = action_requests[key]
                sample_index = len(successes[key])
                adaptive_requests.append(
                    ViabilityRequest(
                        example_index=key[0],
                        fork_id=key[1],
                        action_id=key[2],
                        reference_answer=answer,
                        action_description=description,
                        sample_index=sample_index,
                        seed=viability_trial_seed(
                            args.seed, *key, sample_index
                        ),
                        prompt=prompt,
                    )
                )
            adaptive_outputs = llm.generate(
                [request.prompt for request in adaptive_requests],
                [
                    sampling_params(request.seed)
                    for request in adaptive_requests
                ],
            )
            record_outputs(adaptive_requests, adaptive_outputs)
            total_completions += len(adaptive_requests)
    for record in accepted:
        fork_targets = []
        for raw_fork in record["graph"]["forks"]:
            actions = tuple(GraphAction(**action) for action in raw_fork["actions"])
            sampled_actions = [
                action
                for action in actions
                if action.status not in {"invalid", "dead_end"}
            ]
            action_samples = {
                action.action_id: successes[
                    (
                        int(record["example_index"]),
                        raw_fork["fork_id"],
                        action.action_id,
                    )
                ]
                for action in sampled_actions
            }
            viability = {
                action_id: sum(observations) / len(observations)
                for action_id, observations in action_samples.items()
            }
            fork_targets.append({
                "fork_id": raw_fork["fork_id"],
                "action_ids": [action.action_id for action in actions],
                "viability": viability,
                "samples_by_action": {
                    action_id: len(observations)
                    for action_id, observations in action_samples.items()
                },
                "trial_ids_by_action": {
                    action_id: trial_ids[
                        (
                            int(record["example_index"]),
                            raw_fork["fork_id"],
                            action_id,
                        )
                    ]
                    for action_id in action_samples
                },
                "credible_intervals": {
                    action_id: beta_credible_interval(
                        sum(observations),
                        len(observations),
                        credible_level=args.credible_level,
                    )
                    for action_id, observations in action_samples.items()
                },
                "target": branch_target(actions, viability, temperature=args.temperature),
            })
        append_jsonl(args.output, {
            "schema_version": 1,
            "example_index": record["example_index"],
            "graph_sha256": record["graph_sha256"],
            "fork_targets": fork_targets,
            "samples_per_action": args.samples_per_action,
            "adaptive_max_samples_per_action": adaptive_max,
            "credible_level": args.credible_level,
            "temperature": args.temperature,
            "max_completion_tokens": args.max_completion_tokens,
            "forced_prefix_protocol": action_protocol,
            "model": args.model,
            "model_revision": args.model_revision,
        })
    manifest = {
        "schema_version": 1,
        "graph_cache_sha256": graph_manifest["cache_sha256"],
        "viability_cache": str(args.output),
        "viability_cache_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "trial_cache": str(trials_output),
        "trial_cache_sha256": hashlib.sha256(
            trials_output.read_bytes()
        ).hexdigest(),
        "trial_records": sum(len(values) for values in trial_ids.values()),
        "trial_schema_version": 1,
        "legacy_grader": "generation_common.grade_math@v1",
        "examples": len(accepted),
        "samples_per_action": args.samples_per_action,
        "adaptive_max_samples_per_action": adaptive_max,
        "credible_level": args.credible_level,
        "total_forced_completions": total_completions,
        "temperature": args.temperature,
        "max_completion_tokens": args.max_completion_tokens,
        "forced_prefix_protocol": action_protocol,
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
