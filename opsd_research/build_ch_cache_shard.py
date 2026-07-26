"""Build one data-parallel shard of a contrastive-hindsight OPSD cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from .ch_cache import (
    auditor_prompt,
    blind_student_prompt,
    build_dynamic_dossier_record,
)
from .generation_common import extract_last_boxed, output_token_ids
from .training_data import DATASET_REVISION, load_math_cot_20k


DEFAULT_MODEL = "Qwen/Qwen3-4B"
DEFAULT_MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
MAX_MODEL_LEN = 16384
BLIND_MAX_TOKENS = 2048
AUDIT_MAX_TOKENS = 1536


def _chat_prompt(
    tokenizer: Any,
    text: str,
    *,
    enable_thinking: bool,
    completion_tokens: int,
) -> str:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    if prompt_tokens + completion_tokens > MAX_MODEL_LEN:
        raise ValueError(
            f"full prompt requires {prompt_tokens + completion_tokens} tokens, "
            f"exceeding CH context {MAX_MODEL_LEN}"
        )
    return prompt


def _rejected_record(
    *,
    index: int,
    problem: str,
    reference_solution: str,
    blind_attempts: int,
    shard_id: int,
    num_shards: int,
    reason: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "accepted": False,
        "example_index": index,
        "problem_sha256": hashlib.sha256(problem.encode("utf-8")).hexdigest(),
        "reference_solution_sha256": hashlib.sha256(
            reference_solution.encode("utf-8")
        ).hexdigest(),
        "blind_attempt_count": blind_attempts,
        "builder_model": DEFAULT_MODEL,
        "builder_model_revision": DEFAULT_MODEL_REVISION,
        "training_dataset_revision": DATASET_REVISION,
        "shard_id": shard_id,
        "num_shards": num_shards,
        "rejection_reason": reason,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", required=True, type=int)
    parser.add_argument("--blind-attempts", type=int, default=3)
    parser.add_argument("--audit-attempts", type=int, default=3)
    parser.add_argument("--shard-id", required=True, type=int)
    parser.add_argument("--num-shards", required=True, type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to alter existing CH shard {args.output}")
    if args.limit < 1 or args.blind_attempts not in {2, 3} or args.audit_attempts < 1:
        raise SystemExit("invalid CH cache limits or attempt counts")
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise SystemExit("shard-id must be in [0, num-shards)")
    if args.model != DEFAULT_MODEL or args.model_revision != DEFAULT_MODEL_REVISION:
        raise SystemExit("CH cache generation is pinned to Qwen3-4B@1cfa9a7")

    from vllm import LLM, SamplingParams

    rows = load_math_cot_20k(heldout_fraction=0.0)["train"].select(
        range(args.limit)
    )
    source_indices = [
        index for index in range(args.limit) if index % args.num_shards == args.shard_id
    ]
    llm = LLM(
        model=args.model,
        revision=args.model_revision,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        enable_prefix_caching=True,
    )
    tokenizer = llm.get_tokenizer()
    records: dict[int, dict[str, Any]] = {}
    eligible: list[int] = []
    blind_prompts: list[str] = []
    blind_params: list[Any] = []
    blind_keys: list[tuple[int, int]] = []
    for index in source_indices:
        problem = str(rows[index]["question"])
        reference = str(rows[index]["response"])
        if extract_last_boxed(reference) is None:
            records[index] = _rejected_record(
                index=index,
                problem=problem,
                reference_solution=reference,
                blind_attempts=args.blind_attempts,
                shard_id=args.shard_id,
                num_shards=args.num_shards,
                reason="reference solution has no boxed answer",
            )
            continue
        try:
            rendered = [
                _chat_prompt(
                    tokenizer,
                    blind_student_prompt(problem, attempt_index=attempt),
                    enable_thinking=True,
                    completion_tokens=BLIND_MAX_TOKENS,
                )
                for attempt in range(args.blind_attempts)
            ]
        except ValueError as error:
            records[index] = _rejected_record(
                index=index,
                problem=problem,
                reference_solution=reference,
                blind_attempts=args.blind_attempts,
                shard_id=args.shard_id,
                num_shards=args.num_shards,
                reason=str(error),
            )
            continue
        eligible.append(index)
        for attempt, prompt in enumerate(rendered):
            blind_keys.append((index, attempt))
            blind_prompts.append(prompt)
            blind_params.append(
                SamplingParams(
                    n=1,
                    temperature=1.1,
                    top_p=0.95,
                    top_k=20,
                    max_tokens=BLIND_MAX_TOKENS,
                    seed=args.seed + index * args.blind_attempts + attempt,
                )
            )

    started = time.monotonic()
    blind_outputs = llm.generate(blind_prompts, blind_params, use_tqdm=True)
    attempts_by_index: dict[int, list[str | None]] = {
        index: [None] * args.blind_attempts for index in eligible
    }
    attempt_tokens: dict[int, list[int]] = {
        index: [0] * args.blind_attempts for index in eligible
    }
    for (index, attempt), output in zip(blind_keys, blind_outputs, strict=True):
        generated = output.outputs[0]
        attempts_by_index[index][attempt] = generated.text
        attempt_tokens[index][attempt] = len(output_token_ids(generated))

    pending: set[int] = set(eligible)
    last_errors: dict[int, str] = {}
    for audit_attempt in range(args.audit_attempts):
        if not pending:
            break
        audit_indices: list[int] = []
        audit_prompts: list[str] = []
        audit_params: list[Any] = []
        for index in sorted(pending):
            problem = str(rows[index]["question"])
            reference = str(rows[index]["response"])
            attempts = tuple(str(value) for value in attempts_by_index[index])
            try:
                prompt = _chat_prompt(
                    tokenizer,
                    auditor_prompt(
                        problem=problem,
                        reference_solution=reference,
                        reference_answer=str(extract_last_boxed(reference)),
                        attempts=attempts,
                    ),
                    enable_thinking=False,
                    completion_tokens=AUDIT_MAX_TOKENS,
                )
            except ValueError as error:
                last_errors[index] = str(error)
                continue
            audit_indices.append(index)
            audit_prompts.append(prompt)
            audit_params.append(
                SamplingParams(
                    n=1,
                    temperature=0.2,
                    top_p=0.95,
                    max_tokens=AUDIT_MAX_TOKENS,
                    seed=args.seed + 100_000 + audit_attempt * args.limit + index,
                )
            )
        audit_outputs = llm.generate(
            audit_prompts, audit_params, use_tqdm=True
        ) if audit_prompts else []
        for index, output in zip(audit_indices, audit_outputs, strict=True):
            problem = str(rows[index]["question"])
            reference = str(rows[index]["response"])
            attempts = tuple(str(value) for value in attempts_by_index[index])
            generated = output.outputs[0]
            try:
                record = build_dynamic_dossier_record(
                    example_index=index,
                    problem=problem,
                    reference_solution=reference,
                    blind_attempts=attempts,
                    audit_text=generated.text,
                    builder_seed=args.seed,
                )
                record.update(
                    {
                        "builder_model": args.model,
                        "builder_model_revision": args.model_revision,
                        "training_dataset_revision": DATASET_REVISION,
                        "shard_id": args.shard_id,
                        "num_shards": args.num_shards,
                        "blind_attempt_output_tokens": attempt_tokens[index],
                        "audit_output_tokens": len(output_token_ids(generated)),
                        "audit_builder_attempt": audit_attempt + 1,
                    }
                )
                records[index] = record
                pending.remove(index)
            except ValueError as error:
                last_errors[index] = str(error)

    for index in sorted(pending):
        records[index] = _rejected_record(
            index=index,
            problem=str(rows[index]["question"]),
            reference_solution=str(rows[index]["response"]),
            blind_attempts=args.blind_attempts,
            shard_id=args.shard_id,
            num_shards=args.num_shards,
            reason=last_errors.get(index, "auditor did not produce a valid dossier"),
        )
        records[index]["blind_attempt_output_tokens"] = attempt_tokens[index]

    if set(records) != set(source_indices):
        raise RuntimeError("CH shard builder lost source identities")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(records[index], ensure_ascii=False, sort_keys=True) + "\n"
            for index in source_indices
        ),
        encoding="utf-8",
    )
    accepted = sum(bool(record["accepted"]) for record in records.values())
    print(
        json.dumps(
            {
                "event": "ch_cache_shard_complete",
                "shard_id": args.shard_id,
                "num_shards": args.num_shards,
                "requested": len(source_indices),
                "accepted": accepted,
                "rejected": len(source_indices) - accepted,
                "blind_attempts": args.blind_attempts,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "output": str(args.output),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
