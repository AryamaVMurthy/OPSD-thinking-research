"""Build a deterministic, answer-masked GRAF cache before policy training.

The builder has access to a reference solution, but the saved cache never does.
Every candidate is parsed and rejected as a unit if it leaks an answer or fails
the restricted graph schema.  Training consumes only accepted records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .graf_graph import graph_builder_prompt, parse_answer_masked_graph
from .records import append_jsonl


DEFAULT_MODEL = "Qwen/Qwen3-4B"
DEFAULT_MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
TRAINING_DATASET_REVISION = "1435fb21d4fecc8ad4966a26f22a874cf2b527f1"


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an immutable GRAF graph cache")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--limit", required=True, type=int)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.limit < 1:
        raise SystemExit("--limit must be positive")
    if args.model != DEFAULT_MODEL or args.model_revision != DEFAULT_MODEL_REVISION:
        raise SystemExit("GRAF cache building is pinned to Qwen3-4B@1cfa9a7")
    if args.output.exists():
        raise SystemExit(f"refusing to alter existing cache {args.output}")
    if args.manifest.exists():
        raise SystemExit(f"refusing to alter existing manifest {args.manifest}")

    from vllm import LLM, SamplingParams
    from .training_data import load_math_cot_20k

    rows = load_math_cot_20k()["train"].select(range(args.limit))
    llm = LLM(
        model=args.model,
        revision=args.model_revision,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=8192,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        trust_remote_code=True,
    )
    prompts = [graph_builder_prompt(str(row["question"]), str(row["response"])) for row in rows]
    outputs = llm.generate(
        prompts,
        SamplingParams(temperature=0.2, top_p=0.95, max_tokens=1024, seed=args.seed),
    )
    accepted = 0
    rejected = 0
    for index, (row, output) in enumerate(zip(rows, outputs, strict=True)):
        question = str(row["question"])
        response = str(row["response"])
        raw = output.outputs[0].text
        record: dict[str, Any] = {
            "schema_version": 1,
            "example_index": index,
            "problem_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
            "builder_model": args.model,
            "builder_model_revision": args.model_revision,
            "training_dataset_revision": TRAINING_DATASET_REVISION,
            "builder_seed": args.seed,
        }
        try:
            graph = parse_answer_masked_graph(
                _json_object(raw), problem=question, reference_solution=response
            )
            record.update({"accepted": True, "graph": json.loads(graph.canonical_json()), "graph_sha256": graph.sha256})
            accepted += 1
        except (ValueError, json.JSONDecodeError) as error:
            record.update({"accepted": False, "reject_reason": str(error)})
            rejected += 1
        append_jsonl(args.output, record)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "cache": str(args.output),
        "cache_sha256": digest,
        "requested_examples": args.limit,
        "accepted_examples": accepted,
        "rejected_examples": rejected,
        "builder_model": args.model,
        "builder_model_revision": args.model_revision,
        "training_dataset_revision": TRAINING_DATASET_REVISION,
        "builder_seed": args.seed,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if accepted == 0:
        raise SystemExit("cache contained no valid answer-masked graphs")
    print(json.dumps(manifest, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
