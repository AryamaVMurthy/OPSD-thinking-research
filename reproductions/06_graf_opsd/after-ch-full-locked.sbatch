#!/usr/bin/env bash
# Compare the once-only locked AIME25/AIME26 confirmations and issue the
# pre-registered paper-claim decision.
#SBATCH --job-name=after-ch-full-locked
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node10
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module load u22/cuda/12.4
source "${HOME}/OPSD-thinking-research/infra/turing/common.sh"
cd "${PROJECT_SOURCE}"
: "${CH_AIME25_TREATMENT_RUN:?}" "${CH_AIME25_BASELINE_RUN:?}"
: "${CH_AIME26_TREATMENT_RUN:?}" "${CH_AIME26_BASELINE_RUN:?}"

declare -a comparisons=()
for benchmark in aime25 aime26; do
  upper="${benchmark^^}"
  treatment_var="CH_${upper}_TREATMENT_RUN"
  baseline_var="CH_${upper}_BASELINE_RUN"
  treatment="${RESULTS_ROOT}/${!treatment_var}"
  baseline="${RESULTS_ROOT}/${!baseline_var}"
  test -f "${treatment}/summary.json"
  test -f "${baseline}/summary.json"
  comparison="${treatment}/paired-vs-base.json"
  python3 -m opsd_research.compare_math \
    --baseline-input "${baseline}"/generations.shard*.jsonl \
    --baseline-grades "${baseline}/official-grades.jsonl" \
    --treatment-input "${treatment}"/generations.shard*.jsonl \
    --treatment-grades "${treatment}/official-grades.jsonl" \
    --samples-per-problem 12 \
    --output "${comparison}"
  python3 -m opsd_research.graf_development_report \
    --comparison "${comparison}" \
    --candidate-id "ch1-completion-aware-4096-full-${benchmark}" \
    --output "${treatment}/official-${benchmark}-report.md"
  comparisons+=("${comparison}")
done

report_dir="${RESULTS_ROOT}/ch1-completion-aware-4096-full-locked-confirmation"
mkdir -p "${report_dir}"
python3 - "${comparisons[0]}" "${comparisons[1]}" "${report_dir}" <<'PY'
import json, sys
from pathlib import Path

rows = []
for benchmark, path in zip(("aime25", "aime26"), sys.argv[1:3], strict=True):
    comparison = json.load(open(path, encoding="utf-8"))
    delta = float(comparison["avg_at_12"]["delta"])
    lower, upper = comparison["delta_bootstrap_95ci"]["avg_at_12"]
    rows.append({
        "benchmark": benchmark,
        "avg_at_12_delta": delta,
        "delta_ci_lower": lower,
        "delta_ci_upper": upper,
        "positive_paired_ci": lower > 0,
        "comparison": path,
    })
claim = all(row["positive_paired_ci"] for row in rows)
payload = {
    "schema_version": 1,
    "method": "ch1-completion-aware-4096-full",
    "locked_benchmarks": rows,
    "paper_claim_gate_passed": claim,
    "decision": (
        "positive-paper-claim-supported"
        if claim
        else "report-as-development-result-no-positive-paper-claim"
    ),
}
root = Path(sys.argv[3])
(root / "locked-confirmation.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
lines = [
    "# CH1 locked confirmation", "",
    "| Benchmark | Avg@12 delta | Paired 95% CI | Positive CI |",
    "|---|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f"| {row['benchmark'].upper()} | {row['avg_at_12_delta']:+.6f} | "
        f"[{row['delta_ci_lower']:+.6f}, {row['delta_ci_upper']:+.6f}] | "
        f"{row['positive_paired_ci']} |"
    )
lines.extend(["", f"Decision: **{payload['decision']}**.", ""])
(root / "locked-confirmation.md").write_text(
    "\n".join(lines), encoding="utf-8"
)
print(json.dumps({"event": "ch1_locked_confirmation", **payload}, sort_keys=True))
PY
