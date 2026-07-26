#!/usr/bin/env bash
# Produce the official 12-sample paper-scale CH1-vs-untouched AIME24 result.
#SBATCH --job-name=after-ch-full-aime24
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node10
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module load u22/cuda/12.4
source "${HOME}/OPSD-thinking-research/infra/turing/common.sh"
cd "${PROJECT_SOURCE}"
: "${CH_FULL_TREATMENT_RUN:?}" "${CH_FULL_BASELINE_RUN:?}"
treatment="${RESULTS_ROOT}/${CH_FULL_TREATMENT_RUN}"
baseline="${RESULTS_ROOT}/${CH_FULL_BASELINE_RUN}"
test -f "${treatment}/summary.json"
test -f "${baseline}/summary.json"
python3 -m opsd_research.compare_math \
  --baseline-input "${baseline}"/generations.shard*.jsonl \
  --baseline-grades "${baseline}/official-grades.jsonl" \
  --treatment-input "${treatment}"/generations.shard*.jsonl \
  --treatment-grades "${treatment}/official-grades.jsonl" \
  --samples-per-problem 12 \
  --output "${treatment}/paired-vs-base.json"
python3 -m opsd_research.graf_development_report \
  --comparison "${treatment}/paired-vs-base.json" \
  --candidate-id ch1-completion-aware-4096-full \
  --output "${treatment}/official-aime24-report.md"
python3 - "${treatment}/paired-vs-base.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1], encoding="utf-8"))
d = float(c["avg_at_12"]["delta"])
lo, hi = c["delta_bootstrap_95ci"]["avg_at_12"]
print(json.dumps({
    "event": "ch1_full_official_aime24_result",
    "avg_at_12_delta": d,
    "delta_ci_lower": lo,
    "delta_ci_upper": hi,
    "locked_evaluation_gate": (
        "eligible" if d > 0 else "stop-no-positive-full-scale-effect"
    ),
}, sort_keys=True))
PY
