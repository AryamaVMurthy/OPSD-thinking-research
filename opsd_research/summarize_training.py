from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any

from .run_report import write_training_report


PROGRESS_RE = re.compile(r"(?P<step>\d+)/(?P<total>\d+)")
VLLM_RE = re.compile(
    r"vLLM generation done .*?total tokens: (?P<tokens>\d+), "
    r"avg length: (?P<average>[0-9.]+)"
)
LOSS_RE = re.compile(r"(\{'loss':.*?\})")
STEP_FILE_RE = re.compile(r"generations_step_(\d+)\.json$")
CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)$")


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _parse_training_log(path: Path) -> tuple[list[dict[str, Any]], list[int]]:
    losses: list[dict[str, Any]] = []
    rollout_tokens: list[int] = []
    current_step: int | None = None
    for fragment in re.split(r"[\r\n]+", path.read_text(encoding="utf-8")):
        progress = PROGRESS_RE.search(fragment)
        if progress:
            current_step = int(progress.group("step"))
        generation = VLLM_RE.search(fragment)
        if generation:
            rollout_tokens.append(int(generation.group("tokens")))
        loss_match = LOSS_RE.search(fragment)
        if loss_match:
            parsed = ast.literal_eval(loss_match.group(1))
            if not isinstance(parsed, dict):
                raise ValueError(f"unexpected loss record in {path}")
            record = {"step": current_step, **parsed}
            losses.append(record)
    # A resumed run appends to the canonical log and may replay updates after
    # its last durable checkpoint. Keep the final record for each completed
    # optimizer step while retaining step-less diagnostics verbatim.
    by_step: dict[int, dict[str, Any]] = {}
    without_step: list[dict[str, Any]] = []
    for record in losses:
        step = record.get("step")
        if isinstance(step, int):
            by_step[step] = record
        else:
            without_step.append(record)
    deduplicated = without_step + [by_step[step] for step in sorted(by_step)]
    return deduplicated, rollout_tokens


def _parse_finod_events(path: Path) -> list[dict[str, Any]]:
    events = []
    marker = re.compile(r'\{\s*"event"\s*:\s*"finod_loss"')
    for fragment in re.split(r"[\r\n]+", path.read_text(encoding="utf-8")):
        match = marker.search(fragment)
        if match is None:
            continue
        try:
            event = json.loads(fragment[match.start():])
        except json.JSONDecodeError as error:
            raise ValueError(f"malformed FiNOD event in {path}") from error
        if event.get("event") != "finod_loss":
            raise ValueError(f"unexpected FiNOD event in {path}")
        events.append(event)
    return events


def _parse_gpu_telemetry(path: Path) -> list[dict[str, Any]]:
    by_gpu: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) != 8:
                raise ValueError(f"malformed GPU telemetry row in {path}: {row!r}")
            gpu = int(row[1].strip())
            utilization = float(row[3].strip())
            memory_mib = int(row[4].strip())
            power_watts = float(row[6].strip())
            temperature_c = int(row[7].strip())
            stats = by_gpu.setdefault(
                gpu,
                {
                    "gpu": gpu,
                    "samples": 0,
                    "utilization_sum": 0.0,
                    "max_memory_mib": 0,
                    "max_power_watts": 0.0,
                    "max_temperature_c": 0,
                },
            )
            stats["samples"] += 1
            stats["utilization_sum"] += utilization
            stats["max_memory_mib"] = max(stats["max_memory_mib"], memory_mib)
            stats["max_power_watts"] = max(
                stats["max_power_watts"], power_watts
            )
            stats["max_temperature_c"] = max(
                stats["max_temperature_c"], temperature_c
            )

    result = []
    for gpu, stats in sorted(by_gpu.items()):
        result.append(
            {
                "gpu": gpu,
                "samples": stats["samples"],
                "mean_utilization_percent": (
                    stats["utilization_sum"] / stats["samples"]
                ),
                "max_memory_mib": stats["max_memory_mib"],
                "max_power_watts": stats["max_power_watts"],
                "max_temperature_c": stats["max_temperature_c"],
            }
        )
    return result


def summarize(
    training_dir: Path,
    training_log: Path,
    *,
    max_completion_length: int,
    telemetry: Path | None = None,
) -> dict[str, Any]:
    generation_files = sorted(
        (training_dir / "generations").glob("generations_step_*.json"),
        key=lambda path: int(STEP_FILE_RE.search(path.name).group(1)),
    )
    generations: list[dict[str, Any]] = []
    generation_steps: list[int] = []
    generation_file_records: list[dict[str, Any]] = []
    for path in generation_files:
        match = STEP_FILE_RE.search(path.name)
        if match is None:
            continue
        step = int(match.group(1))
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("generations")
        if payload.get("step") != step or not isinstance(rows, list):
            raise ValueError(f"malformed generation dump: {path}")
        generation_steps.append(step)
        generations.extend(rows)
        generation_file_records.append(
            {"step": step, "path": str(path), "samples": len(rows)}
        )

    completions = [str(row.get("completion", "")) for row in generations]
    losses, rollout_tokens = _parse_training_log(training_log)
    finod_events = _parse_finod_events(training_log)

    checkpoint_records = []
    for path in sorted(
        training_dir.glob("checkpoint-*"),
        key=lambda item: int(CHECKPOINT_RE.search(item.name).group(1)),
    ):
        match = CHECKPOINT_RE.search(path.name)
        if match is None:
            continue
        adapter = path / "adapter_model.safetensors"
        trainer_state = path / "trainer_state.json"
        checkpoint_records.append(
            {
                "step": int(match.group(1)),
                "path": str(path),
                "bytes": _directory_bytes(path),
                "adapter_present": adapter.is_file(),
                "trainer_state_present": trainer_state.is_file(),
            }
        )

    latest_loss = losses[-1] if losses else None
    latest_generation_step = max(generation_steps) if generation_steps else None
    latest_checkpoint_step = (
        max(record["step"] for record in checkpoint_records)
        if checkpoint_records
        else None
    )
    summary = {
        "schema_version": 1,
        "training_dir": str(training_dir),
        "training_log": str(training_log),
        "latest_logged_step": latest_loss["step"] if latest_loss else None,
        "loss_history": losses,
        "loss_finite": all(
            isinstance(record.get("loss"), (int, float))
            and float("-inf") < float(record["loss"]) < float("inf")
            for record in losses
        ),
        "finod_signal_history": finod_events,
        "finod_signal": {
            "events": len(finod_events),
            "all_finite": all(
                all(
                    isinstance(event.get(key), (int, float))
                    and float("-inf") < float(event[key]) < float("inf")
                    for key in (
                        "loss",
                        "guide_energy",
                        "nuisance_energy",
                        "residual_energy",
                        "target_kl",
                    )
                )
                for event in finod_events
            ),
            "positive_loss_events": sum(
                float(event.get("loss", 0.0)) > 0.0
                for event in finod_events
            ),
            "max_target_kl": max(
                (
                    float(
                        event.get(
                            "max_observed_target_kl",
                            event["target_kl"],
                        )
                    )
                    for event in finod_events
                ),
                default=None,
            ),
            "mean_residual_energy": (
                mean(float(event["residual_energy"]) for event in finod_events)
                if finod_events
                else None
            ),
            "mean_collapsed_residual_fraction": (
                mean(
                    float(event["collapsed_residual_fraction"])
                    for event in finod_events
                )
                if finod_events
                else None
            ),
        },
        "checkpoints": checkpoint_records,
        "generation_dumps": generation_file_records,
        "rollout_dump_integrity": {
            "latest_generation_step": latest_generation_step,
            "latest_checkpoint_step": latest_checkpoint_step,
            "covers_latest_checkpoint": (
                latest_generation_step is not None
                and latest_checkpoint_step is not None
                and latest_generation_step >= latest_checkpoint_step
            ),
        },
        "recorded_rollouts": {
            "count": len(completions),
            "nonempty": sum(bool(text.strip()) for text in completions),
            "thinking_started": sum("<think>" in text for text in completions),
            "thinking_closed": sum("</think>" in text for text in completions),
            "boxed_answer": sum("\\boxed" in text for text in completions),
        },
        "vllm_rollout_calls": {
            "count": len(rollout_tokens),
            "mean_tokens": mean(rollout_tokens) if rollout_tokens else None,
            "min_tokens": min(rollout_tokens) if rollout_tokens else None,
            "max_tokens": max(rollout_tokens) if rollout_tokens else None,
            "at_completion_cap": sum(
                tokens == max_completion_length for tokens in rollout_tokens
            ),
            "completion_cap": max_completion_length,
        },
        "gpu_telemetry": _parse_gpu_telemetry(telemetry) if telemetry else None,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dir", required=True, type=Path)
    parser.add_argument("--training-log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--telemetry", type=Path)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    args = parser.parse_args()

    summary = summarize(
        args.training_dir,
        args.training_log,
        max_completion_length=args.max_completion_length,
        telemetry=args.telemetry,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_training_report(summary, args.output.parent)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
