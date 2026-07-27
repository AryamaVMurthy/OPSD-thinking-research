#!/usr/bin/env bash
# Compare Fluid-JS to the immutable paired base; never auto-touch locked sets.
#SBATCH --job-name=after-fluid-js-eval
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
: "${FLUID_EVAL_RUN:?}"

baseline="${RESULTS_ROOT}/graf-aime24-base-full-context-screen-20260727"
treatment="${RESULTS_ROOT}/${FLUID_EVAL_RUN}"
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
  --candidate-id fluid-g4-js-128 \
  --output "${treatment}/development-report.md"
python3 - "${treatment}/paired-vs-base.json" <<'PY'
import json
import sys

comparison = json.load(open(sys.argv[1], encoding="utf-8"))
delta = comparison["avg_at_4"]["delta"]
lower, upper = comparison["delta_bootstrap_95ci"]["avg_at_4"]
print(json.dumps({
    "event": "fluid_g4_development_result",
    "avg_at_4_delta": delta,
    "delta_ci_lower": lower,
    "delta_ci_upper": upper,
    "locked_benchmarks_used": False,
    "next_gate": (
        "matched-context/control-and-second-seed"
        if delta > 0
        else "diagnose-before-more-training"
    ),
}, sort_keys=True))
PY
