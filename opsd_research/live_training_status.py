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
# Dataset preprocessing has its own ``19428/19428 ... examples/s`` bar.  A
# trainer update bar carries an iteration duration, so only that form is a
# valid observed optimizer step.
_PROGRESS = re.compile(r"(?P<step>\d+)/(?:\d+)\s+\[[^\]]*s/it\]")


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
    branch_events = []
    forward_kl_values = []
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
        kl_start = line.find('{"event":"exact_forward_kl_loss"')
        if kl_start >= 0:
            kl_end = line.find("}", kl_start)
            if kl_end >= 0:
                try:
                    event = json.loads(line[kl_start : kl_end + 1])
                except json.JSONDecodeError:
                    event = None
                value = _finite_number(event.get("value")) if isinstance(event, dict) else None
                if value is not None:
                    forward_kl_values.append(value)
        branch_start = line.find('{"event":"graf_branch_loss"')
        if branch_start >= 0:
            branch_end = line.find("}", branch_start)
            if branch_end >= 0:
                try:
                    event = json.loads(line[branch_start : branch_end + 1])
                except json.JSONDecodeError:
                    event = None
                if isinstance(event, dict):
                    active = _finite_number(event.get("active_forks"))
                    branch_kl = _finite_number(event.get("branch_kl"))
                    entropy_floor = _finite_number(event.get("entropy_floor"))
                    weighted_loss = _finite_number(event.get("weighted_loss"))
                    if None not in (active, branch_kl, entropy_floor, weighted_loss):
                        branch_events.append({
                            "active_forks": active,
                            "branch_kl": branch_kl,
                            "entropy_floor": entropy_floor,
                            "weighted_loss": weighted_loss,
                        })
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
    branch_active = [row for row in branch_events if row["active_forks"] > 0]
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
        "graf_branch": {
            # This hook runs once per routed microbatch (including gradient
            # accumulation), not once per optimizer update.
            "loss_calls": len(branch_events),
            "active_loss_calls": len(branch_active),
            "active_loss_call_rate": len(branch_active) / len(branch_events) if branch_events else None,
            "mean_active_forks_per_loss_call": (
                sum(row["active_forks"] for row in branch_events) / len(branch_events)
                if branch_events else None
            ),
            "mean_branch_kl_when_active": (
                sum(row["branch_kl"] for row in branch_active) / len(branch_active)
                if branch_active else None
            ),
            "mean_entropy_floor_when_active": (
                sum(row["entropy_floor"] for row in branch_active) / len(branch_active)
                if branch_active else None
            ),
            "mean_weighted_loss_when_active": (
                sum(row["weighted_loss"] for row in branch_active) / len(branch_active)
                if branch_active else None
            ),
        },
        "forward_kl": {
            "loss_calls": len(forward_kl_values),
            "min": min(forward_kl_values) if forward_kl_values else None,
            "max": max(forward_kl_values) if forward_kl_values else None,
            "mean": (
                sum(forward_kl_values) / len(forward_kl_values)
                if forward_kl_values else None
            ),
            "negative_loss_calls": sum(value < 0.0 for value in forward_kl_values),
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
    branch = status["graf_branch"]
    forward_kl = status["forward_kl"]
    if forward_kl["loss_calls"]:
        lines.extend([
            "",
            "## Exact forward-KL telemetry",
            "",
            f"- Loss calls: `{forward_kl['loss_calls']}`",
            f"- Mean / min / max: `{forward_kl['mean']:.10g}` / `{forward_kl['min']:.10g}` / `{forward_kl['max']:.10g}`",
            f"- Negative loss calls: `{forward_kl['negative_loss_calls']}`",
        ])
    if branch["loss_calls"]:
        lines.extend([
            "",
            "## GRAF routing activity",
            "",
            f"- Branch-active loss calls: `{100 * branch['active_loss_call_rate']:.1f}%` ({branch['active_loss_calls']}/{branch['loss_calls']})",
            f"- Mean active forks/loss call: `{branch['mean_active_forks_per_loss_call']:.2f}`",
            f"- Mean branch KL (active loss calls): `{branch['mean_branch_kl_when_active']:.6f}`",
            f"- Mean entropy-floor term (active loss calls): `{branch['mean_entropy_floor_when_active']:.6f}`",
            f"- Mean weighted routing loss (active loss calls): `{branch['mean_weighted_loss_when_active']:.6f}`",
        ])
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
