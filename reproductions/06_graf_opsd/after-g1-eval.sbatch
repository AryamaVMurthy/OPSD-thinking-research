#!/usr/bin/env bash
#SBATCH --job-name=graf-after-g1-eval
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node10
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module load u22/cuda/12.4
source "${HOME}/OPSD-thinking-research/infra/turing/common.sh"
cd "${PROJECT_SOURCE}"

: "${G1_PILOT_JOB_ID:?missing G1 pilot job ID}"
: "${G1_EVAL_RUN:?missing G1 evaluation run name}"
baseline="${RESULTS_ROOT}/graf-aime24-base-20260726"
treatment="${RESULTS_ROOT}/${G1_EVAL_RUN}"
test -f "${baseline}/summary.json"
test -f "${treatment}/summary.json"

python3 -m opsd_research.compare_math \
  --baseline-input "${baseline}"/generations.shard*.jsonl \
  --baseline-grades "${baseline}/official-grades.jsonl" \
  --treatment-input "${treatment}"/generations.shard*.jsonl \
  --treatment-grades "${treatment}/official-grades.jsonl" \
  --samples-per-problem 12 \
  --output "${treatment}/paired-vs-base.json"
python3 - "${treatment}/paired-vs-base.json" <<'PY'
import json
import sys

comparison = json.load(open(sys.argv[1], encoding="utf-8"))
delta = comparison["avg_at_12"]["delta"]
lower, upper = comparison["delta_bootstrap_95ci"]["avg_at_12"]
print(json.dumps({
    "event": "g1_development_result",
    "candidate_id": "g1-scaffold-2048",
    "avg_at_12_delta": delta,
    "delta_ci_lower": lower,
    "delta_ci_upper": upper,
    "promotion_gate_passed": delta >= 0.03 and lower > 0 and upper > 0,
}, sort_keys=True))
PY
