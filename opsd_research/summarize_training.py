from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any


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
    return losses, rollout_tokens


def summarize(
    training_dir: Path,
    training_log: Path,
    *,
    max_completion_length: int,
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
        "checkpoints": checkpoint_records,
        "generation_dumps": generation_file_records,
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
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dir", required=True, type=Path)
    parser.add_argument("--training-log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    args = parser.parse_args()

    summary = summarize(
        args.training_dir,
        args.training_log,
        max_completion_length=args.max_completion_length,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
