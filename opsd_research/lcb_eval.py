from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from .config import load_config
from .generation_common import finish_reason, output_token_ids, split_thinking
from .lcb_data import load_lcb_v6, upstream_path
from .prompts import lcb_messages, render_thinking_prompt
from .records import append_jsonl, key, prompt_hash, read_jsonl, stable_seed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--method", default="qwen3-instruct")
    parser.add_argument("--checkpoint", default="none")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config(args.config).data
    if config["kind"] != "lcb_eval":
        raise SystemExit("lcb_eval requires kind=lcb_eval")
    if not 0 <= args.shard_id < args.num_shards:
        raise SystemExit("shard-id must be in [0, num-shards)")

    sys.path.insert(0, str(upstream_path()))
    from lcb_runner.lm_styles import LMStyle
    from lcb_runner.utils.extraction_utils import extract_code

    benchmark = load_lcb_v6(config["dataset_revision"])
    if len(benchmark) != config["expected_count"]:
        raise RuntimeError(
            f"LCB {config['release_version']}: expected {config['expected_count']}, "
            f"found {len(benchmark)}"
        )
    if args.limit:
        benchmark = benchmark[: args.limit]

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

    requests = []
    prompts = []
    params = []
    for problem_index, problem in enumerate(benchmark):
        prompt = render_thinking_prompt(
            tokenizer,
            lcb_messages(problem.question_content, problem.starter_code),
        )
        prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        effective_max_tokens = min(
            int(config["max_new_tokens"]),
            int(config["max_model_len"]) - prompt_tokens,
        )
        if effective_max_tokens <= 0:
            raise RuntimeError(
                f"problem {problem.question_id} prompt has {prompt_tokens} tokens, "
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
                "livecodebench-v6-thinking",
                str(problem.question_id),
                sample_index,
            )
            if record_key in existing:
                continue
            seed = stable_seed(
                int(config["base_seed"]),
                config["model"],
                args.method,
                str(args.checkpoint),
                "livecodebench-v6-thinking",
                str(problem.question_id),
                sample_index,
            )
            requests.append(
                (
                    problem,
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
                "benchmark": "livecodebench-v6-thinking",
                "model": config["model"],
                "method": args.method,
                "checkpoint": str(args.checkpoint),
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
            problem,
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
        code = extract_code(text, LMStyle.CodeQwenInstruct)
        record = {
            "schema_version": 1,
            "model": config["model"],
            "model_revision": config["model_revision"],
            "method": args.method,
            "checkpoint": str(args.checkpoint),
            "benchmark": "livecodebench-v6-thinking",
            "dataset_revision": config["dataset_revision"],
            "problem_id": str(problem.question_id),
            "sample_index": sample_index,
            "seed": seed,
            "enable_thinking": True,
            "prompt_hash": prompt_hash(prompt),
            "prompt_tokens": prompt_tokens,
            "requested_max_new_tokens": config["max_new_tokens"],
            "effective_max_new_tokens": effective_max_tokens,
            "question_title": problem.question_title,
            "problem": problem.question_content,
            "response": text,
            "reasoning": reasoning,
            "final": final,
            "code": code,
            "code_extracted": bool(code.strip()),
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
                    "problem_id": str(problem.question_id),
                    "sample_index": sample_index,
                    "code_extracted": bool(code.strip()),
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
