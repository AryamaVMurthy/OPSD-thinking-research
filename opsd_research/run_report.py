"""Human-readable, auditable reports for training and benchmark runs.

The raw JSON summaries remain the source of truth.  This module intentionally
derives reports from those summaries instead of maintaining a second metrics
path, so an attractive Markdown report can never silently disagree with the
machine-readable artifact used by promotion gates.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import yaml


REPORT_SCHEMA_VERSION = 1


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _key_values(path: Path) -> dict[str, str]:
    text = _read_text(path)
    if text is None:
        return {}
    result: dict[str, str] = {}
    for line in text.splitlines():
        key, separator, value = line.partition("=")
        if separator and key:
            result[key] = value
    return result


def _config(run_dir: Path) -> dict[str, Any]:
    text = _read_text(run_dir / "config.yaml")
    if text is None:
        return {}
    loaded = yaml.safe_load(text)
    return loaded if isinstance(loaded, dict) else {}


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return f"{value:.{digits}g}"
    return str(value)


def _table(rows: list[tuple[str, Any]]) -> list[str]:
    return ["| Field | Value |", "|---|---|"] + [
        f"| {key} | {_fmt(value)} |" for key, value in rows
    ]


def _sparkline(values: list[float]) -> str:
    """Return a compact trend trace without a plotting dependency."""
    if len(values) < 2:
        return "insufficient points"
    finite = [value for value in values if math.isfinite(value)]
    if len(finite) != len(values):
        return "contains non-finite values"
    lower, upper = min(values), max(values)
    if upper == lower:
        return "▁" * len(values)
    glyphs = "▁▂▃▄▅▆▇█"
    return "".join(glyphs[round((value - lower) / (upper - lower) * (len(glyphs) - 1))] for value in values)


def _artifact_names(run_dir: Path) -> list[str]:
    interesting = (
        "config.yaml", "runtime-overrides.txt", "source-commit.txt",
        "environment-freeze.txt", "training-summary.json", "summary.json",
        "official-grades.jsonl", "paired-vs-base.json",
    )
    found = [name for name in interesting if (run_dir / name).is_file()]
    found.extend(sorted(path.name for path in run_dir.glob("manifest-*.sha256")))
    found.extend(sorted(path.name for path in run_dir.glob("artifact-manifest-*.sha256")))
    return sorted(set(found))


def training_report(summary: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Build a complete training report from the already persisted summary."""
    losses = [
        float(item["loss"])
        for item in summary.get("loss_history", [])
        if isinstance(item.get("loss"), (int, float))
    ]
    grad_norms = [
        float(item["grad_norm"])
        for item in summary.get("loss_history", [])
        if isinstance(item.get("grad_norm"), (int, float))
    ]
    rollout = dict(summary.get("vllm_rollout_calls") or {})
    configured_cap = rollout.get("completion_cap")
    rollout_count = int(rollout.get("count") or 0)
    at_cap = int(rollout.get("at_completion_cap") or 0)
    config = _config(run_dir)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_type": "training",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "source_commit": _read_text(run_dir / "source-commit.txt"),
        "runtime": _key_values(next(iter(run_dir.glob("runtime-*.txt")), run_dir / "runtime.txt")),
        "runtime_overrides": _key_values(run_dir / "runtime-overrides.txt"),
        "config": config,
        "optimization": {
            "loss_points": len(losses),
            "initial_loss": losses[0] if losses else None,
            "final_loss": losses[-1] if losses else None,
            "min_loss": min(losses) if losses else None,
            "max_loss": max(losses) if losses else None,
            "loss_delta": losses[-1] - losses[0] if len(losses) > 1 else None,
            "loss_sparkline": _sparkline(losses),
            "finite": bool(summary.get("loss_finite")),
            "mean_grad_norm": mean(grad_norms) if grad_norms else None,
            "max_grad_norm": max(grad_norms) if grad_norms else None,
        },
        "rollouts": {
            **rollout,
            "completion_cap_rate": at_cap / rollout_count if rollout_count else None,
            "configured_completion_cap": configured_cap,
            "recorded": summary.get("recorded_rollouts"),
        },
        "checkpoint_integrity": summary.get("rollout_dump_integrity"),
        "gpu_telemetry": summary.get("gpu_telemetry"),
        "artifacts": _artifact_names(run_dir),
    }


def evaluation_report(summary: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Build a report for a benchmark run scored by the official harness."""
    sample_count = int(summary.get("samples_per_problem") or 0)
    prefix = f"avg_at_{sample_count}" if sample_count else "avg_at_12"
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_type": "evaluation",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "source_commit": _read_text(run_dir / "source-commit.txt"),
        "runtime": _key_values(next(iter(run_dir.glob("runtime-*.txt")), run_dir / "runtime.txt")),
        "config": _config(run_dir),
        "identity": {
            key: summary.get(key)
            for key in ("model", "model_revision", "method", "checkpoint", "benchmark", "dataset_revision")
        },
        "metrics": {
            prefix: summary.get(prefix),
            f"pass_at_{sample_count}": summary.get(f"pass_at_{sample_count}"),
            f"maj_at_{sample_count}": summary.get(f"maj_at_{sample_count}"),
            "bootstrap_95ci": summary.get("bootstrap_95ci"),
            "official_format_rate": summary.get("official_format_rate"),
            "boxed_format_rate": summary.get("boxed_format_rate"),
            "nonempty_thinking_rate": summary.get("nonempty_thinking_rate"),
            "length_cutoff_rate": summary.get("length_cutoff_rate"),
            "output_tokens": summary.get("output_tokens"),
        },
        "official_protocol": {
            key: summary.get(key)
            for key in ("scoring_protocol", "grader_revision", "competition_config", "competition_config_sha256", "strict_parsing")
        },
        "artifacts": _artifact_names(run_dir),
    }


def _training_markdown(report: dict[str, Any]) -> str:
    config = report["config"]
    opt = report["optimization"]
    rollouts = report["rollouts"]
    lines = [
        f"# Training report: `{Path(report['run_dir']).name}`", "",
        "## Provenance", "",
        *_table([
            ("Source commit", report.get("source_commit")),
            ("Candidate / variant", config.get("variant")),
            ("Model", config.get("model")),
            ("Model revision", config.get("model_revision")),
            ("Dataset revision", config.get("dataset_revision")),
            ("Node", report.get("runtime", {}).get("node")),
            ("Slurm job", report.get("runtime", {}).get("slurm_job_id")),
        ]), "",
        "## Method and optimization", "",
        *_table([
            ("Graph mode", config.get("graph_mode")),
            ("Completion cap", config.get("max_completion_length")),
            ("Max steps", config.get("max_steps")),
            ("Effective batch", config.get("effective_batch_size")),
            ("Learning rate", config.get("learning_rate")),
            ("LoRA r / alpha", f"{config.get('lora_r')} / {config.get('lora_alpha')}"),
            ("Teacher", "fixed privileged teacher" if config.get("fixed_teacher") else "not fixed"),
            ("Objective", "full-vocabulary teacher→student KL (beta=0), clipped"),
        ]), "",
        "## Optimization trace", "",
        *_table([
            ("Finite losses", opt.get("finite")),
            ("Loss points", opt.get("loss_points")),
            ("Initial → final", f"{_fmt(opt.get('initial_loss'))} → {_fmt(opt.get('final_loss'))}"),
            ("Loss delta", opt.get("loss_delta")),
            ("Minimum / maximum", f"{_fmt(opt.get('min_loss'))} / {_fmt(opt.get('max_loss'))}"),
            ("Mean / max grad norm", f"{_fmt(opt.get('mean_grad_norm'))} / {_fmt(opt.get('max_grad_norm'))}"),
        ]), "",
        f"Loss trace (each point is one logged optimizer update): `{opt.get('loss_sparkline')}`", "",
        "## Rollouts and integrity", "",
        *_table([
            ("vLLM calls", rollouts.get("count")),
            ("Mean / max output tokens", f"{_fmt(rollouts.get('mean_tokens'))} / {_fmt(rollouts.get('max_tokens'))}"),
            ("At completion cap", f"{rollouts.get('at_completion_cap')} ({_fmt(rollouts.get('completion_cap_rate'))})"),
            ("Rollout dump covers checkpoint", (report.get("checkpoint_integrity") or {}).get("covers_latest_checkpoint")),
            ("Recorded completions", (rollouts.get("recorded") or {}).get("count")),
            ("Closed thinking / boxed", f"{(rollouts.get('recorded') or {}).get('thinking_closed')} / {(rollouts.get('recorded') or {}).get('boxed_answer')}"),
        ]), "",
        "## GPU telemetry", "",
        "| GPU | Mean util. | Max memory MiB | Max temperature °C |", "|---:|---:|---:|---:|",
    ]
    for gpu in report.get("gpu_telemetry") or []:
        lines.append(
            f"| {gpu.get('gpu')} | {_fmt(gpu.get('mean_utilization_percent'))} | "
            f"{_fmt(gpu.get('max_memory_mib'))} | {_fmt(gpu.get('max_temperature_c'))} |"
        )
    lines.extend(["", "## Artifacts", ""])
    lines.extend(f"- `{name}`" for name in report["artifacts"])
    return "\n".join(lines) + "\n"


def _evaluation_markdown(report: dict[str, Any]) -> str:
    identity = report["identity"]
    metrics = report["metrics"]
    lines = [
        f"# Evaluation report: `{Path(report['run_dir']).name}`", "",
        "## Identity", "",
        *_table([(key.replace("_", " ").title(), value) for key, value in identity.items()]), "",
        "## Official benchmark metrics", "",
        *_table([(key.replace("_", " "), value) for key, value in metrics.items()]), "",
        "## Official scoring provenance", "",
        *_table([(key.replace("_", " "), value) for key, value in report["official_protocol"].items()]), "",
        "## Artifacts", "",
    ]
    lines.extend(f"- `{name}`" for name in report["artifacts"])
    return "\n".join(lines) + "\n"


def write_training_report(summary: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    report = training_report(summary, run_dir)
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "report.md").write_text(_training_markdown(report), encoding="utf-8")
    return report


def write_evaluation_report(summary: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    report = evaluation_report(summary, run_dir)
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "report.md").write_text(_evaluation_markdown(report), encoding="utf-8")
    return report
