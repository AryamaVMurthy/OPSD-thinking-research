from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from .config import load_config
from .lcb_data import load_lcb_v6, upstream_path
from .records import read_jsonl, validate_unique_complete


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--input", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--graded-output", required=True, type=Path)
    parser.add_argument("--processes", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=6)
    args = parser.parse_args()
    config = load_config(args.config).data
    records = [record for path in args.input for record in read_jsonl(path)]
    problem_ids = sorted({str(record["problem_id"]) for record in records})
    if len(problem_ids) != config["expected_count"]:
        raise ValueError(
            f"expected {config['expected_count']} LCB problems, observed {len(problem_ids)}"
        )
    validate_unique_complete(records, problem_ids, config["samples_per_problem"])

    sys.path.insert(0, str(upstream_path()))
    from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics

    benchmark = load_lcb_v6(config["dataset_revision"])
    by_problem = defaultdict(list)
    for record in records:
        by_problem[str(record["problem_id"])].append(record)
    generations = []
    samples = []
    ordered_records = []
    for problem in benchmark:
        problem_records = sorted(
            by_problem[str(problem.question_id)], key=lambda item: item["sample_index"]
        )
        ordered_records.append(problem_records)
        generations.append([record["code"] for record in problem_records])
        samples.append(problem.get_evaluation_sample())

    metrics, results, metadata = codegen_metrics(
        samples,
        generations,
        k_list=[1, 5],
        num_process_evaluate=args.processes,
        timeout=args.timeout,
    )
    args.graded_output.parent.mkdir(parents=True, exist_ok=True)
    with args.graded_output.open("w", encoding="utf-8") as handle:
        for index, problem_records in enumerate(ordered_records):
            for sample_index, record in enumerate(problem_records):
                graded = all(value is True for value in results[index][sample_index])
                enriched = dict(record)
                enriched["correct"] = graded
                enriched["execution_metadata"] = metadata[index][sample_index]
                handle.write(json.dumps(enriched, ensure_ascii=False, sort_keys=True) + "\n")

    lengths = sorted(int(record["output_tokens"]) for record in records)
    summary = {
        "schema_version": 1,
        "model": records[0]["model"],
        "method": records[0]["method"],
        "checkpoint": records[0]["checkpoint"],
        "benchmark": "livecodebench-v6-thinking",
        "num_problems": len(benchmark),
        "samples_per_problem": config["samples_per_problem"],
        "pass_at_1": metrics["pass@1"],
        "pass_at_5": metrics["pass@5"],
        "code_extraction_rate": mean(bool(record["code_extracted"]) for record in records),
        "nonempty_thinking_rate": mean(
            bool(record["nonempty_thinking"]) for record in records
        ),
        "length_cutoff_rate": mean(
            record["finish_reason"] == "length" for record in records
        ),
        "output_tokens": {
            "mean": mean(lengths),
            "median": median(lengths),
            "p90": lengths[int(0.9 * (len(lengths) - 1))],
            "max": max(lengths),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
