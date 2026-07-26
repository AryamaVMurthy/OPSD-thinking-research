"""Generate an auditable, partial-progress report from a running training log."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any


_ROLLOUT = re.compile(
    r"vLLM generation done - elapsed time: (?P<seconds>[0-9.]+)s, prompts: "
    r"(?P<prompts>\d+), total tokens: (?P<tokens>\d+)"
)
_PROGRESS = re.compile(r"(?P<step>\d+)/(?:\d+)")


def _finite_number(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def summarize_log(path: Path, max_completion_length: int) -> dict[str, Any]:
    if max_completion_length <= 0:
        raise ValueError("max_completion_length must be positive")
    text = path.read_text(encoding="utf-8", errors="replace")
    rollouts = []
    losses = []
    observed_step = 0
    for match in _ROLLOUT.finditer(text):
        rollouts.append({
            "elapsed_seconds": float(match.group("seconds")),
            "prompts": int(match.group("prompts")),
            "tokens": int(match.group("tokens")),
        })
    for match in _PROGRESS.finditer(text):
        observed_step = max(observed_step, int(match.group("step")))
    # HF logs Python dicts, one per rank.  Only accept complete, literal dicts
    # carrying an actual loss to avoid interpreting arbitrary log fragments.
    for line in text.splitlines():
        start = line.find("{'loss':")
        if start < 0:
            continue
        end = line.find("}", start)
        if end < 0:
            continue
        try:
            event = ast.literal_eval(line[start : end + 1])
        except (SyntaxError, ValueError):
            continue
        loss = _finite_number(event.get("loss")) if isinstance(event, dict) else None
        grad = _finite_number(event.get("grad_norm")) if isinstance(event, dict) else None
        if loss is not None:
            losses.append({"loss": loss, "grad_norm": grad})
    token_total = sum(row["tokens"] for row in rollouts)
    elapsed_total = sum(row["elapsed_seconds"] for row in rollouts)
    capped = sum(row["tokens"] >= max_completion_length for row in rollouts)
    return {
        "schema_version": 1,
        "log": str(path),
        "observed_optimizer_steps": observed_step,
        "loss_history": losses,
        "rollouts": {
            "calls": len(rollouts),
            "token_total": token_total,
            "mean_tokens": token_total / len(rollouts) if rollouts else None,
            "capped_calls": capped,
            "cap_rate": capped / len(rollouts) if rollouts else None,
            "mean_generation_seconds": elapsed_total / len(rollouts) if rollouts else None,
            "tokens_per_second": token_total / elapsed_total if elapsed_total else None,
        },
    }


def render_markdown(status: dict[str, Any]) -> str:
    rollouts = status["rollouts"]
    lines = [
        "# Live training status",
        "",
        f"- Observed optimizer steps: `{status['observed_optimizer_steps']}`",
        f"- Rollout calls: `{rollouts['calls']}`",
        f"- Mean rollout tokens: `{rollouts['mean_tokens']:.1f}`" if rollouts["mean_tokens"] is not None else "- Mean rollout tokens: `n/a`",
        f"- Completion cap rate: `{100 * rollouts['cap_rate']:.1f}%` ({rollouts['capped_calls']}/{rollouts['calls']})" if rollouts["cap_rate"] is not None else "- Completion cap rate: `n/a`",
        f"- Generation throughput: `{rollouts['tokens_per_second']:.1f} tokens/s`" if rollouts["tokens_per_second"] is not None else "- Generation throughput: `n/a`",
    ]
    losses = status["loss_history"]
    if losses:
        lines.extend(["", "## Optimizer metrics", "", "| Update | Loss | Gradient norm |", "|---:|---:|---:|"])
        for index, event in enumerate(losses, start=1):
            grad = "n/a" if event["grad_norm"] is None else f"{event['grad_norm']:.6f}"
            lines.append(f"| {index} | {event['loss']:.6f} | {grad} |")
    else:
        lines.extend(["", "No optimizer update has been logged yet."])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-log", required=True, type=Path)
    parser.add_argument("--max-completion-length", required=True, type=int)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    args = parser.parse_args()
    status = summarize_log(args.training_log, args.max_completion_length)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = render_markdown(status)
    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(markdown, encoding="utf-8")
    print(markdown, end="")


if __name__ == "__main__":
    main()
