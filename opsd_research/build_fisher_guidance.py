"""Build independent procedural plans from problem text alone."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .build_graf_cache import (
    DEFAULT_MODEL,
    DEFAULT_MODEL_REVISION,
    TRAINING_DATASET_REVISION,
)
from .fisher_guidance import (
    CACHE_KIND,
    GUIDANCE_INPUT_PROTOCOL,
    validate_answer_free_plan,
    validate_guidance_ensemble,
)


MAX_MODEL_LEN = 8192
MAX_PLAN_TOKENS = 320


def _plan_prompt(tokenizer: Any, problem: str, plan_index: int) -> str:
    """Render a non-thinking request whose only data input is the problem."""
    lenses = (
        "Choose the most natural structural representation and decisive checks.",
        "Seek a genuinely different invariant, decomposition, or coordinate system.",
        "Stress-test likely shortcuts and give a robust route with a recovery step.",
    )
    instruction = (
        "Write one concise, high-level procedural plan for the math problem "
        "below. Do not solve it. Do not calculate or state any intermediate "
        "or final numerical value, conclusion, boxed expression, or option. "
        "Do not use a numbered list and do not repeat the problem. Describe "
        "the representation to introduce, the key deductions to attempt, "
        "checks that would falsify the route, and a fallback if it fails. "
        "Output only one paragraph of three to five sentences.\n\n"
        f"Planning lens: {lenses[plan_index % len(lenses)]}\n\n"
        f"Problem:\n{problem}"
    )
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if "</think>" not in rendered:
        raise RuntimeError("plan builder chat template did not disable thinking")
    return rendered


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--limit", required=True, type=int)
    parser.add_argument("--plans-per-problem", type=int, default=3)
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--selection-seed", type=int, default=73)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--data-source",
        action="append",
        dest="data_sources",
        default=[],
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    return parser.parse_args()


def _select_indices(rows, *, limit: int, seed: int) -> list[int]:
    ranked = []
    for index, row in enumerate(rows):
        payload = json.dumps(
            {
                "question": str(row["question"]),
                "data_source": str(row["data_source"]),
                "seed": int(seed),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        ranked.append(
            (hashlib.sha256(payload.encode("utf-8")).hexdigest(), index)
        )
    if not 1 <= limit <= len(ranked):
        raise ValueError("guidance limit exceeds eligible problem count")
    return [index for _digest, index in sorted(ranked)[:limit]]


def main() -> None:
    args = _parse_args()
    if not 2 <= args.plans_per_problem <= 5:
        raise SystemExit("plans-per-problem must be in [2, 5]")
    if args.attempts < 1:
        raise SystemExit("attempts must be positive")
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise SystemExit("shard-id must be in [0, num-shards)")
    if args.model != DEFAULT_MODEL or args.model_revision != DEFAULT_MODEL_REVISION:
        raise SystemExit("guidance building is pinned to Qwen3-4B@1cfa9a7")
    if args.output.exists() or args.manifest.exists():
        raise SystemExit("refusing to overwrite a guidance cache")

    from vllm import LLM, SamplingParams

    from .training_data import load_math_cot_questions_only

    # PyArrow projects only these columns while reading the source Parquet;
    # the response is never materialized in this process.
    rows = load_math_cot_questions_only()
    rows = rows.add_column("_source_index", list(range(len(rows))))
    requested_sources = set(args.data_sources)
    if requested_sources:
        rows = rows.filter(
            lambda row: str(row["data_source"]) in requested_sources,
            desc="Selecting problem-only guidance sources",
        )
    global_indices = _select_indices(
        rows, limit=args.limit, seed=args.selection_seed
    )
    shard_local_indices = global_indices[
        args.shard_id :: args.num_shards
    ]
    selected = rows.select(shard_local_indices)

    llm = LLM(
        model=args.model,
        revision=args.model_revision,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.86,
        enforce_eager=True,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()
    plans: list[list[str | None]] = [
        [None] * args.plans_per_problem for _ in range(len(selected))
    ]
    errors: dict[tuple[int, int], str] = {}
    for plan_index in range(args.plans_per_problem):
        pending = list(range(len(selected)))
        for attempt in range(args.attempts):
            if not pending:
                break
            prompts = [
                _plan_prompt(
                    tokenizer,
                    str(selected[row_index]["question"]).strip(),
                    plan_index,
                )
                for row_index in pending
            ]
            outputs = llm.generate(
                prompts,
                SamplingParams(
                    temperature=0.75,
                    top_p=0.92,
                    top_k=40,
                    max_tokens=MAX_PLAN_TOKENS,
                    seed=args.seed + 1000 * plan_index + attempt,
                ),
            )
            retry: list[int] = []
            for row_index, output in zip(pending, outputs, strict=True):
                problem = str(selected[row_index]["question"]).strip()
                candidate = output.outputs[0].text.strip()
                try:
                    validate_answer_free_plan(problem, candidate)
                    plans[row_index][plan_index] = candidate
                except ValueError as error:
                    errors[(row_index, plan_index)] = str(error)
                    retry.append(row_index)
            pending = retry

    args.output.parent.mkdir(parents=True, exist_ok=True)
    accepted = 0
    rejected = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for row_index, row_plans in enumerate(plans):
            problem = str(selected[row_index]["question"]).strip()
            if any(plan is None for plan in row_plans):
                rejected += 1
                continue
            complete_plans = [str(plan) for plan in row_plans]
            try:
                validate_guidance_ensemble(problem, complete_plans)
            except ValueError as error:
                errors[(row_index, -1)] = str(error)
                rejected += 1
                continue
            source_index = int(selected[row_index]["_source_index"])
            record = {
                "source_index": source_index,
                "problem": problem,
                "problem_sha256": hashlib.sha256(
                    problem.encode("utf-8")
                ).hexdigest(),
                "plans": complete_plans,
                "seeds": [
                    args.seed + 1000 * pair
                    for pair in range(args.plans_per_problem)
                ],
            }
            handle.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            )
            accepted += 1
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "cache_kind": CACHE_KIND,
        "answer_access": False,
        "reference_solution_access": False,
        "guidance_input_protocol": GUIDANCE_INPUT_PROTOCOL,
        "plans_per_problem": args.plans_per_problem,
        "accepted_records": accepted,
        "rejected_records": rejected,
        "requested_records": len(selected),
        "requested_global_records": args.limit,
        "records_file": args.output.name,
        "records_sha256": digest,
        "builder_model": args.model,
        "builder_model_revision": args.model_revision,
        "training_dataset_revision": TRAINING_DATASET_REVISION,
        "data_sources": sorted(requested_sources),
        "selection_seed": args.selection_seed,
        "builder_seed": args.seed,
        "builder_attempts": args.attempts,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "rejection_reasons": {
            f"{row}:{pair}": reason
            for (row, pair), reason in sorted(errors.items())
            if pair == -1 or plans[row][pair] is None
        },
    }
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
