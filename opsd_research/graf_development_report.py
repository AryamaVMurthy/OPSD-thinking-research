"""Render a compact, auditable human report for a GRAF development result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _metric_key(comparison: dict[str, Any]) -> str:
    sample_count = comparison.get("samples_per_problem")
    key = f"avg_at_{sample_count}"
    if not isinstance(sample_count, int) or key not in comparison:
        raise ValueError("comparison does not contain its declared Avg@n metric")
    return key


def render(comparison: dict[str, Any], *, candidate_id: str) -> str:
    """Render paired official evidence without applying a CI gate."""
    key = _metric_key(comparison)
    metric = comparison[key]
    interval = comparison["delta_bootstrap_95ci"][key]
    changes = comparison["paired_sample_changes"]
    required = ("baseline", "treatment", "delta")
    if any(name not in metric for name in required) or len(interval) != 2:
        raise ValueError("malformed paired comparison")
    delta = float(metric["delta"])
    decision = "promote to full confirmation" if delta > 0 else "advance to next candidate"
    return "\n".join([
        "# GRAF development result", "",
        f"- Candidate: `{candidate_id}`",
        f"- Benchmark: `{comparison['benchmark']}`",
        f"- Model: `{comparison['model']}`",
        f"- Protocol: `{comparison['comparison_protocol']}`",
        f"- Paired problems / samples per problem: `{comparison['num_problems']}` / `{comparison['samples_per_problem']}`", "",
        "## Official paired result", "",
        "| Metric | Baseline | Candidate | Delta | Paired bootstrap 95% CI |",
        "|---|---:|---:|---:|---:|",
        f"| {key} | {100 * float(metric['baseline']):.2f}% | {100 * float(metric['treatment']):.2f}% | {100 * delta:+.2f} pp | [{100 * float(interval[0]):+.2f}, {100 * float(interval[1]):+.2f}] pp |", "",
        "## Sample-level transitions", "",
        f"- Improved: `{changes['improved']}`; degraded: `{changes['degraded']}`",
        f"- Both correct: `{changes['both_correct']}`; both wrong: `{changes['both_wrong']}`", "",
        "## Controller decision", "",
        f"`{decision}` — policy uses the observed {key} delta ({100 * delta:+.2f} pp); bootstrap uncertainty is logged for interpretation only.", "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
    report = render(comparison, candidate_id=args.candidate_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
