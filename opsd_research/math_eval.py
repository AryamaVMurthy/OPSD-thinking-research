from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from .config import MATH_DATASETS, load_config
from .generation_common import (
    extract_last_boxed,
    finish_reason,
    grade_math,
    output_token_ids,
    split_thinking,
)
from .prompts import math_messages, render_thinking_prompt
from .records import (
    append_jsonl,
    key,
    prompt_hash,
    read_jsonl,
    stable_seed,
    validate_adapter_identity,
)


def _field(row: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in row:
            return row[name]
    raise KeyError(f"none of {names!r} found in dataset row fields {sorted(row)}")


def _load_rows(config: dict[str, Any], limit: int | None) -> list[dict[str, str]]:
    from datasets import load_dataset

    alias = config["dataset"]
    spec = MATH_DATASETS[alias]
    dataset = load_dataset(
        spec["path"],
        split=spec["split"],
        revision=config["dataset_revision"],
        trust_remote_code=True,
    )
    if len(dataset) != spec["expected_count"]:
        raise RuntimeError(
            f"{alias}: expected {spec['expected_count']} rows, found {len(dataset)}"
        )
    rows: list[dict[str, str]] = []
    for index, raw in enumerate(dataset):
        problem = str(_field(raw, ("problem", "question", "prompt")))
        answer = str(_field(raw, ("answer", "solution", "ground_truth")))
        problem_id = str(
            raw.get("problem_idx", raw.get("id", raw.get("question_id", index)))
        )
        rows.append(
            {"problem_id": problem_id, "problem": problem, "answer": answer}
        )
    return rows[:limit] if limit else rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--adapter-sha256")
    parser.add_argument("--method", default="qwen3-instruct")
    parser.add_argument("--checkpoint", default="none")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config(args.config).data
    if config["kind"] != "math_eval":
        raise SystemExit("math_eval requires kind=math_eval")
    if not 0 <= args.shard_id < args.num_shards:
        raise SystemExit("shard-id must be in [0, num-shards)")
    try:
        validate_adapter_identity(args.adapter, args.adapter_sha256)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    rows = _load_rows(config, args.limit)

    existing = set()
    if args.output.exists():
        existing = {key(record) for record in read_jsonl(args.output)}

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"],
        revision=config["model_revision"],
        trust_remote_code=True,
    )
    llm_kwargs: dict[str, Any] = {
        "model": config["model"],
        "revision": config["model_revision"],
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": float(config.get("gpu_memory_utilization", 0.90)),
        "max_model_len": int(config.get("max_model_len", 40960)),
        "enforce_eager": bool(config.get("enforce_eager", True)),
    }
    if args.adapter:
        llm_kwargs.update(
            enable_lora=True,
            max_lora_rank=64,
            max_loras=1,
            max_cpu_loras=1,
        )
    llm = LLM(**llm_kwargs)

    lora_request = None
    if args.adapter:
        adapter_file = args.adapter / "adapter_model.safetensors"
        if not adapter_file.exists():
            raise FileNotFoundError(f"missing LoRA adapter: {adapter_file}")
        from vllm.lora.request import LoRARequest

        lora_request = LoRARequest("opsd", 1, str(args.adapter.resolve()))

    requests: list[tuple[dict[str, str], int, str, int, int, int]] = []
    prompts: list[str] = []
    params: list[Any] = []
    for problem_index, row in enumerate(rows):
        prompt = render_thinking_prompt(tokenizer, math_messages(row["problem"]))
        prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        effective_max_tokens = min(
            int(config["max_new_tokens"]),
            int(config["max_model_len"]) - prompt_tokens,
        )
        if effective_max_tokens <= 0:
            raise RuntimeError(
                f"problem {row['problem_id']} prompt has {prompt_tokens} tokens, "
                f"exceeding max_model_len={config['max_model_len']}"
            )
        for sample_index in range(config["samples_per_problem"]):
            flat_index = problem_index * config["samples_per_problem"] + sample_index
            if flat_index % args.num_shards != args.shard_id:
                continue
            record_key = (
                config["model"],
                args.method,
                str(args.checkpoint),
                config["dataset"],
                row["problem_id"],
                sample_index,
            )
            if record_key in existing:
                continue
            seed = stable_seed(
                int(config["base_seed"]),
                config["model"],
                args.method,
                str(args.checkpoint),
                config["dataset"],
                row["problem_id"],
                sample_index,
            )
            requests.append(
                (
                    row,
                    sample_index,
                    prompt,
                    seed,
                    prompt_tokens,
                    effective_max_tokens,
                )
            )
            prompts.append(prompt)
            params.append(
                SamplingParams(
                    n=1,
                    temperature=config["temperature"],
                    top_p=config["top_p"],
                    top_k=config["top_k"],
                    min_p=config["min_p"],
                    presence_penalty=config["presence_penalty"],
                    max_tokens=effective_max_tokens,
                    seed=seed,
                )
            )

    print(
        json.dumps(
            {
                "event": "generation_start",
                "benchmark": config["dataset"],
                "model": config["model"],
                "method": args.method,
                "checkpoint": str(args.checkpoint),
                "adapter_sha256": args.adapter_sha256,
                "shard_id": args.shard_id,
                "num_shards": args.num_shards,
                "requests": len(requests),
                "thinking": True,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not requests:
        print('{"event":"nothing_to_generate"}')
        return

    started = time.monotonic()
    outputs = llm.generate(
        prompts,
        params,
        lora_request=lora_request,
        use_tqdm=True,
    )
    for ordinal, (
        (
            row,
            sample_index,
            prompt,
            seed,
            prompt_tokens,
            effective_max_tokens,
        ),
        request_output,
    ) in enumerate(
        zip(requests, outputs), start=1
    ):
        generated = request_output.outputs[0]
        text = generated.text
        reasoning, final, nonempty_thinking = split_thinking(text)
        predicted = extract_last_boxed(text)
        correct = grade_math(predicted, row["answer"])
        record = {
            "schema_version": 1,
            "model": config["model"],
            "model_revision": config["model_revision"],
            "method": args.method,
            "checkpoint": str(args.checkpoint),
            "adapter_sha256": args.adapter_sha256,
            "benchmark": config["dataset"],
            "dataset_revision": config["dataset_revision"],
            "problem_id": row["problem_id"],
            "sample_index": sample_index,
            "seed": seed,
            "enable_thinking": True,
            "prompt_hash": prompt_hash(prompt),
            "prompt_tokens": prompt_tokens,
            "requested_max_new_tokens": config["max_new_tokens"],
            "effective_max_new_tokens": effective_max_tokens,
            "problem": row["problem"],
            "ground_truth": row["answer"],
            "response": text,
            "reasoning": reasoning,
            "final": final,
            "predicted_answer": predicted,
            "formatted": predicted is not None,
            "correct": correct,
            "nonempty_thinking": nonempty_thinking,
            "output_tokens": len(output_token_ids(generated)),
            "finish_reason": finish_reason(generated),
        }
        append_jsonl(args.output, record)
        print(
            json.dumps(
                {
                    "event": "sample_complete",
                    "ordinal": ordinal,
                    "total": len(requests),
                    "problem_id": row["problem_id"],
                    "sample_index": sample_index,
                    "correct": correct,
                    "formatted": predicted is not None,
                    "nonempty_thinking": nonempty_thinking,
                    "tokens": record["output_tokens"],
                    "finish_reason": record["finish_reason"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    print(
        json.dumps(
            {
                "event": "generation_complete",
                "samples": len(requests),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "output": str(args.output),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
