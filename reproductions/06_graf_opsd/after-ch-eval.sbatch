#!/usr/bin/env bash
# Produce the paired CH-vs-untouched screening result without auto-promoting.
#SBATCH --job-name=after-ch-eval
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node03
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module load u22/cuda/12.4
source "${HOME}/OPSD-thinking-research/infra/turing/common.sh"
cd "${PROJECT_SOURCE}"
: "${CH_EVAL_RUN:?}"
: "${CH_CANDIDATE_ID:?}"
baseline="${RESULTS_ROOT}/graf-aime24-base-full-context-screen-20260727"
treatment="${RESULTS_ROOT}/${CH_EVAL_RUN}"
test -f "${baseline}/summary.json"
test -f "${treatment}/summary.json"
python3 -m opsd_research.compare_math \
  --baseline-input "${baseline}"/generations.shard*.jsonl \
  --baseline-grades "${baseline}/official-grades.jsonl" \
  --treatment-input "${treatment}"/generations.shard*.jsonl \
  --treatment-grades "${treatment}/official-grades.jsonl" \
  --samples-per-problem 4 \
  --output "${treatment}/paired-vs-base.json"
python3 -m opsd_research.graf_development_report \
  --comparison "${treatment}/paired-vs-base.json" \
  --candidate-id "${CH_CANDIDATE_ID}" \
  --output "${treatment}/development-report.md"
python3 - "${treatment}/paired-vs-base.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1], encoding="utf-8"))
d = c["avg_at_4"]["delta"]
lo, hi = c["delta_bootstrap_95ci"]["avg_at_4"]
print(json.dumps({
    "event": "ch_development_result",
    "avg_at_4_delta": d,
    "delta_ci_lower": lo,
    "delta_ci_upper": hi,
    "next_gate": "50-step/1000-row confirmation" if d > 0 else "diagnose-before-more-GPU",
}, sort_keys=True))
PY
