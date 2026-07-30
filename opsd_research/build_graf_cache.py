"""Build a deterministic, answer-masked GRAF cache before policy training.

The builder has access to a reference solution, but the saved cache never does.
Every candidate is parsed and rejected as a unit if it leaks an answer or fails
the restricted graph schema.  Training consumes only accepted records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .answer_masking import (
    ANSWER_LEAKAGE_PROTOCOL,
    PROBLEM_ONLY_GUIDANCE_PROTOCOL,
)
from .finod_dataset import finod_training_row_from_graph
from .graf_graph import (
    graph_builder_prompt,
    graph_critic_prompt,
    graph_sanitizer_prompt,
    parse_answer_masked_graph,
)
from .records import append_jsonl


DEFAULT_MODEL = "Qwen/Qwen3-4B"
DEFAULT_MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
TRAINING_DATASET_REVISION = "1435fb21d4fecc8ad4966a26f22a874cf2b527f1"
MAX_MODEL_LEN = 40960
MAX_COMPLETION_TOKENS = 1024
CHAT_TEMPLATE_RESERVE_TOKENS = 256


def _representative_source_indices(
    rows: Sequence[dict[str, Any]],
    *,
    limit: int,
    seed: int,
    shard_id: int = 0,
    num_shards: int = 1,
) -> list[int]:
    """Select a reproducible uniform content-hash sample and one balanced shard."""
    if not 1 <= limit <= len(rows):
        raise ValueError("representative limit must be in [1, number of rows]")
    if num_shards < 1 or not 0 <= shard_id < num_shards:
        raise ValueError("shard_id must be in [0, num_shards)")
    ranked = []
    for index, row in enumerate(rows):
        payload = json.dumps(
            {
                "question": str(row["question"]),
                "response": str(row["response"]),
                "selection_seed": int(seed),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        ranked.append(
            (hashlib.sha256(payload.encode("utf-8")).hexdigest(), index)
        )
    selected = [index for _digest, index in sorted(ranked)[:limit]]
    return selected[shard_id::num_shards]


def _json_object(text: str) -> dict[str, Any]:
    """Extract one JSON object, tolerating a Markdown code fence from a builder."""
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.split("\n", 1)[1] if "\n" in candidate else ""
        candidate = candidate.rsplit("```", 1)[0].strip()
    start, end = candidate.find("{"), candidate.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("builder response did not contain a JSON object")
    payload = json.loads(candidate[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("builder JSON must be an object")
    return payload


def _bounded_builder_prompt(
    tokenizer: Any, problem: str, reference_solution: str, *, graph_budget: int = 24
) -> str:
    """Keep a cache-builder request inside its declared context window.

    The builder is offline and may inspect the reference, whereas the saved
    graph is still answer-masked by ``parse_answer_masked_graph``. Preserve as
    much of an unusually long reference as fits, and label any truncation
    rather than letting one example abort the whole immutable cache build.
    """
    max_prompt_tokens = (
        MAX_MODEL_LEN - MAX_COMPLETION_TOKENS - CHAT_TEMPLATE_RESERVE_TOKENS
    )
    prompt = graph_builder_prompt(problem, reference_solution, graph_budget=graph_budget)
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    if len(prompt_tokens) <= max_prompt_tokens:
        return prompt

    reference_tokens = tokenizer.encode(reference_solution, add_special_tokens=False)
    overhead = len(prompt_tokens) - len(reference_tokens)
    keep = max(0, max_prompt_tokens - overhead - 32)
    while True:
        truncated_reference = tokenizer.decode(
            reference_tokens[:keep], skip_special_tokens=True
        ) + "\n[Reference truncated for the builder context window.]"
        prompt = graph_builder_prompt(
            problem, truncated_reference, graph_budget=graph_budget
        )
        if len(tokenizer.encode(prompt, add_special_tokens=False)) <= max_prompt_tokens:
            return prompt
        if keep == 0:
            raise ValueError("problem and graph-builder instructions exceed the context window")
        keep = max(0, int(keep * 0.9))


def _builder_chat_prompt(
    tokenizer: Any, problem: str, reference_solution: str, *, graph_budget: int = 24
) -> str:
    """Render a non-thinking JSON compiler turn for the offline graph builder.

    Thinking remains enabled for student rollouts and benchmark evaluation. The
    builder instead uses Qwen's documented disabled-thinking template because
    its sole product is a compact JSON object; this prevents a long hidden
    chain-of-thought from consuming the fixed 1,024-token JSON budget or being
    accidentally retained in an answer-masked artifact.
    """
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": _bounded_builder_prompt(tokenizer, problem, reference_solution, graph_budget=graph_budget)}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if "</think>" not in prompt:
        raise RuntimeError("builder chat template did not disable thinking")
    if len(tokenizer.encode(prompt, add_special_tokens=False)) > MAX_MODEL_LEN - MAX_COMPLETION_TOKENS:
        raise ValueError("builder chat prompt exceeds its context window")
    return prompt


def _sanitizer_chat_prompt(
    tokenizer: Any, problem: str, *, graph_budget: int = 24
) -> str:
    """Render a reference-free non-thinking JSON rewrite turn."""
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": graph_sanitizer_prompt(problem, graph_budget=graph_budget)}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if "</think>" not in prompt:
        raise RuntimeError("sanitizer chat template did not disable thinking")
    if len(tokenizer.encode(prompt, add_special_tokens=False)) > MAX_MODEL_LEN - MAX_COMPLETION_TOKENS:
        raise ValueError("sanitizer chat prompt exceeds its context window")
    return prompt


def _cache_builder_chat_prompt(
    tokenizer: Any,
    problem: str,
    reference_solution: str,
    *,
    answer_blind: bool,
    graph_budget: int = 24,
) -> str:
    """Select an auditable problem-only or legacy privileged builder input."""
    if answer_blind:
        return _sanitizer_chat_prompt(
            tokenizer, problem, graph_budget=graph_budget
        )
    return _builder_chat_prompt(
        tokenizer,
        problem,
        reference_solution,
        graph_budget=graph_budget,
    )


def _critic_chat_prompt(
    tokenizer: Any, problem: str, reference_solution: str, candidate_graph: dict[str, Any], *, graph_budget: int = 24
) -> str:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": graph_critic_prompt(
            problem, reference_solution, candidate_graph, graph_budget=graph_budget
        )}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    if "</think>" not in prompt:
        raise RuntimeError("critic chat template did not disable thinking")
    if len(tokenizer.encode(prompt, add_special_tokens=False)) > MAX_MODEL_LEN - MAX_COMPLETION_TOKENS:
        raise ValueError("critic chat prompt exceeds its context window")
    return prompt


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an immutable GRAF graph cache")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--limit", required=True, type=int)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-forks", type=int, default=6)
    parser.add_argument("--max-actions-per-fork", type=int, default=6)
    parser.add_argument("--graph-budget", type=int, default=24)
    parser.add_argument("--teacher-critique", action="store_true")
    parser.add_argument(
        "--answer-blind",
        action="store_true",
        help="generate every guide from the problem alone; references are audit-only",
    )
    parser.add_argument("--selection-seed", type=int)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if (
        args.limit < 1 or args.attempts < 1 or args.max_forks < 1
        or args.max_actions_per_fork < 2 or args.graph_budget < 2
    ):
        raise SystemExit("cache limits, graph action limits, and graph budget must be positive")
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise SystemExit("shard-id must be in [0, num-shards)")
    if args.model != DEFAULT_MODEL or args.model_revision != DEFAULT_MODEL_REVISION:
        raise SystemExit("GRAF cache building is pinned to Qwen3-4B@1cfa9a7")
    if args.answer_blind and args.teacher_critique:
        raise SystemExit("answer-blind cache building forbids privileged teacher critique")
    if args.output.exists():
        raise SystemExit(f"refusing to alter existing cache {args.output}")
    if args.manifest.exists():
        raise SystemExit(f"refusing to alter existing manifest {args.manifest}")

    from vllm import LLM, SamplingParams
    from .training_data import load_math_cot_20k

    all_rows = load_math_cot_20k()["train"]
    if args.limit > len(all_rows):
        raise SystemExit(
            f"cache limit {args.limit} exceeds dataset size {len(all_rows)}"
        )
    if args.selection_seed is None:
        source_indices = list(range(args.limit))[
            args.shard_id :: args.num_shards
        ]
        selection_protocol = "source-prefix-v1"
    else:
        source_indices = _representative_source_indices(
            all_rows,
            limit=args.limit,
            seed=args.selection_seed,
            shard_id=args.shard_id,
            num_shards=args.num_shards,
        )
        selection_protocol = "content-hash-uniform-v1"
    rows = all_rows.select(source_indices)
    llm = LLM(
        model=args.model,
        revision=args.model_revision,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()
    records: list[dict[str, Any] | None] = [None] * len(rows)
    pending: list[int] = list(range(len(rows)))
    accepted = 0
    last_errors: dict[int, str] = {}
    for attempt in range(args.attempts):
        if not pending:
            break
        prompts: list[str] = []
        viable_indices: list[int] = []
        for index in pending:
            row = rows[index]
            try:
                prompts.append(_cache_builder_chat_prompt(
                    tokenizer,
                    str(row["question"]),
                    str(row["response"]),
                    answer_blind=args.answer_blind,
                    graph_budget=args.graph_budget,
                ))
                viable_indices.append(index)
            except ValueError as error:
                last_errors[index] = str(error)
        outputs = llm.generate(
            prompts,
            SamplingParams(
                temperature=0.35,
                top_p=0.95,
                max_tokens=MAX_COMPLETION_TOKENS,
                seed=args.seed + attempt,
            ),
        ) if prompts else []
        next_pending: list[int] = []
        sanitizer_inputs: list[tuple[int, dict[str, Any]]] = []
        for index, output in zip(viable_indices, outputs, strict=True):
            row = rows[index]
            question = str(row["question"])
            response = str(row["response"])
            record: dict[str, Any] = {
                "schema_version": 1,
                "example_index": source_indices[index],
                "problem_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
                "builder_model": args.model,
                "builder_model_revision": args.model_revision,
                "training_dataset_revision": TRAINING_DATASET_REVISION,
                "builder_seed": args.seed + attempt,
                "builder_attempt": attempt + 1,
                "guidance_input_protocol": (
                    PROBLEM_ONLY_GUIDANCE_PROTOCOL
                    if args.answer_blind
                    else "problem-plus-reference-filtered-v1"
                ),
                "answer_leakage_protocol": ANSWER_LEAKAGE_PROTOCOL,
            }
            try:
                graph = parse_answer_masked_graph(
                    _json_object(output.outputs[0].text),
                    problem=question,
                    reference_solution=response,
                    max_forks=args.max_forks,
                    max_actions_per_fork=args.max_actions_per_fork,
                    graph_budget=args.graph_budget,
                    check_reference_fragments=not args.answer_blind,
                )
                if args.answer_blind:
                    finod_training_row_from_graph(
                        question=question,
                        reference_solution=response,
                        graph=graph,
                        source_index=source_indices[index],
                    )
                record.update({
                    "accepted": True,
                    "graph": json.loads(graph.canonical_json()),
                    "graph_sha256": graph.sha256,
                })
                records[index] = record
                accepted += 1
            except (ValueError, json.JSONDecodeError) as error:
                last_errors[index] = str(error)
                next_pending.append(index)
                if str(error) == "graph copies a five-word reference fragment":
                    try:
                        candidate_graph = _json_object(output.outputs[0].text)
                        # This validates structure and hard answer leakage, but
                        # deliberately skips only the copied-phrase rule before
                        # the reference-free rewrite below.
                        parse_answer_masked_graph(
                            candidate_graph,
                            problem=question,
                            reference_solution=response,
                            max_forks=args.max_forks,
                            max_actions_per_fork=args.max_actions_per_fork,
                            graph_budget=args.graph_budget,
                            check_reference_fragments=False,
                        )
                        sanitizer_inputs.append((index, candidate_graph))
                    except (ValueError, json.JSONDecodeError):
                        pass
        sanitizer_prompts: list[str] = []
        sanitizer_indices: list[int] = []
        for index, _candidate_graph in sanitizer_inputs:
            row = rows[index]
            try:
                sanitizer_prompts.append(
                    _sanitizer_chat_prompt(
                        tokenizer, str(row["question"]), graph_budget=args.graph_budget
                    )
                )
                sanitizer_indices.append(index)
            except ValueError as error:
                last_errors[index] = str(error)
        sanitized_outputs = llm.generate(
            sanitizer_prompts,
            SamplingParams(
                temperature=0.2,
                top_p=0.95,
                max_tokens=MAX_COMPLETION_TOKENS,
                seed=args.seed + 10_000 + attempt,
            ),
        ) if sanitizer_prompts else []
        for index, output in zip(sanitizer_indices, sanitized_outputs, strict=True):
            row = rows[index]
            question = str(row["question"])
            response = str(row["response"])
            try:
                graph = parse_answer_masked_graph(
                    _json_object(output.outputs[0].text),
                    problem=question,
                    reference_solution=response,
                    max_forks=args.max_forks,
                    max_actions_per_fork=args.max_actions_per_fork,
                    graph_budget=args.graph_budget,
                )
                records[index] = {
                    "schema_version": 1,
                    "example_index": source_indices[index],
                    "problem_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
                    "builder_model": args.model,
                    "builder_model_revision": args.model_revision,
                    "training_dataset_revision": TRAINING_DATASET_REVISION,
                    "builder_seed": args.seed + 10_000 + attempt,
                    "builder_attempt": attempt + 1,
                    "sanitized_reference_free": True,
                    "accepted": True,
                    "graph": json.loads(graph.canonical_json()),
                    "graph_sha256": graph.sha256,
                }
                accepted += 1
            except (ValueError, json.JSONDecodeError) as error:
                last_errors[index] = str(error)
        next_pending = [index for index in next_pending if records[index] is None]
        pending = next_pending
    rejected_indices = [index for index, record in enumerate(records) if record is None]
    rejected = len(rejected_indices)
    for index in rejected_indices:
        row = rows[index]
        question = str(row["question"])
        records[index] = {
            "schema_version": 1,
            "example_index": source_indices[index],
            "problem_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
            "builder_model": args.model,
            "builder_model_revision": args.model_revision,
            "training_dataset_revision": TRAINING_DATASET_REVISION,
            "builder_seed": args.seed,
            "builder_attempts": args.attempts,
            "guidance_input_protocol": (
                PROBLEM_ONLY_GUIDANCE_PROTOCOL
                if args.answer_blind
                else "problem-plus-reference-filtered-v1"
            ),
            "answer_leakage_protocol": ANSWER_LEAKAGE_PROTOCOL,
            "accepted": False,
            "reject_reason": last_errors.get(index, "prompt could not be rendered"),
        }
    critique_applied = 0
    critique_rejected = 0
    if args.teacher_critique:
        critic_prompts: list[str] = []
        critic_indices: list[int] = []
        for index, record in enumerate(records):
            if record is None or not record.get("accepted"):
                continue
            row = rows[index]
            try:
                critic_prompts.append(_critic_chat_prompt(
                    tokenizer, str(row["question"]), str(row["response"]),
                    dict(record["graph"]), graph_budget=args.graph_budget,
                ))
                critic_indices.append(index)
            except ValueError as error:
                record["teacher_critique_error"] = str(error)
                critique_rejected += 1
        critic_outputs = llm.generate(
            critic_prompts,
            SamplingParams(
                temperature=0.2, top_p=0.95, max_tokens=MAX_COMPLETION_TOKENS,
                seed=args.seed + 20_000,
            ),
        ) if critic_prompts else []
        for index, output in zip(critic_indices, critic_outputs, strict=True):
            record = records[index]
            if record is None:
                raise RuntimeError("critic lost an accepted graph record")
            row = rows[index]
            try:
                graph = parse_answer_masked_graph(
                    _json_object(output.outputs[0].text),
                    problem=str(row["question"]), reference_solution=str(row["response"]),
                    max_forks=args.max_forks, max_actions_per_fork=args.max_actions_per_fork,
                    graph_budget=args.graph_budget,
                )
                record["graph"] = json.loads(graph.canonical_json())
                record["graph_sha256"] = graph.sha256
                record["teacher_critique_applied"] = True
                critique_applied += 1
            except (ValueError, json.JSONDecodeError) as error:
                # Preserve the independently validated original graph while
                # recording that no privileged revision was admitted.
                record["teacher_critique_applied"] = False
                record["teacher_critique_error"] = str(error)
                critique_rejected += 1
    for record in records:
        if record is None:
            raise RuntimeError("cache builder lost an example record")
        append_jsonl(args.output, record)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "cache": str(args.output),
        "cache_sha256": digest,
        "requested_examples": len(rows),
        "requested_global_examples": args.limit,
        "selection_protocol": selection_protocol,
        "selection_seed": args.selection_seed,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "accepted_examples": accepted,
        "rejected_examples": rejected,
        "builder_model": args.model,
        "builder_model_revision": args.model_revision,
        "training_dataset_revision": TRAINING_DATASET_REVISION,
        "builder_seed": args.seed,
        "builder_attempts": args.attempts,
        "max_forks": args.max_forks,
        "max_actions_per_fork": args.max_actions_per_fork,
        "graph_budget": args.graph_budget,
        "teacher_critique": args.teacher_critique,
        "teacher_critique_applied": critique_applied,
        "teacher_critique_rejected": critique_rejected,
        "guidance_input_protocol": (
            PROBLEM_ONLY_GUIDANCE_PROTOCOL
            if args.answer_blind
            else "problem-plus-reference-filtered-v1"
        ),
        "answer_leakage_protocol": ANSWER_LEAKAGE_PROTOCOL,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if accepted == 0:
        raise SystemExit("cache contained no valid answer-masked graphs")
    print(json.dumps(manifest, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
