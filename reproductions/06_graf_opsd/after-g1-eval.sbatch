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
: "${CACHE_BUILD_JOB_ID:?missing immutable graph-cache build job ID}"
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
python3 - "${treatment}/paired-vs-base.json" "artifacts/graf_opsd/candidate-ledger.jsonl" <<'PY'
import json
import sys

from pathlib import Path

from opsd_research.graf_contract import append_ledger

comparison = json.load(open(sys.argv[1], encoding="utf-8"))
delta = comparison["avg_at_12"]["delta"]
lower, upper = comparison["delta_bootstrap_95ci"]["avg_at_12"]
event = {
    "event": "g1_development_result",
    "candidate_id": "g1-scaffold-2048",
    "avg_at_12_delta": delta,
    "delta_ci_lower": lower,
    "delta_ci_upper": upper,
    "promotion_gate_passed": delta >= 0.03 and lower > 0 and upper > 0,
    "comparison_protocol": comparison["comparison_protocol"],
}
event["ledger_event_sha256"] = append_ledger(Path(sys.argv[2]), event)
print(json.dumps(event, sort_keys=True))
PY

promote="$(python3 - "${treatment}/paired-vs-base.json" <<'PY'
import json
import sys

comparison = json.load(open(sys.argv[1], encoding="utf-8"))
delta = comparison["avg_at_12"]["delta"]
lower, upper = comparison["delta_bootstrap_95ci"]["avg_at_12"]
print("1" if delta >= 0.03 and lower > 0 and upper > 0 else "0")
PY
)"
if [[ "${promote}" == "1" ]]; then
  adapter="${TRAIN_ROOT}/graf-g1-answer-masked-50-${G1_PILOT_JOB_ID}"
  locked="$(sbatch --parsable --export="ALL,METHOD=g1-answer-masked-scaffold,ADAPTER=${adapter}" \
    reproductions/06_graf_opsd/submit-locked-evals.sbatch)"
  echo "g1_promoted_locked_submit_job=${locked}"
  exit 0
fi

# G1 failed the pre-registered development gate.  Start the next *isolated*
# candidate from immutable graph inputs: first construct forced-continuation
# targets, then only submit training after its own smoke check.
g2_array="$(sbatch --parsable --array=0-7 \
  --export="ALL,CACHE_BUILD_JOB_ID=${CACHE_BUILD_JOB_ID}" \
  reproductions/06_graf_opsd/build-g2-viability-shard.sbatch)"
echo "g2_viability_array_job=${g2_array}"
g2_merge="$(sbatch --parsable --dependency="afterok:${g2_array}" \
  --export="ALL,CACHE_BUILD_JOB_ID=${CACHE_BUILD_JOB_ID},G2_VIABILITY_ARRAY_JOB_ID=${g2_array}" \
  reproductions/06_graf_opsd/merge-g2-viability.sbatch)"
echo "g2_viability_merge_job=${g2_merge}"
graph_manifest="${SCRATCH_ROOT}/graf_cache/g1-${CACHE_BUILD_JOB_ID}/manifest.json"
viability_manifest="${SCRATCH_ROOT}/graf_cache/g2-viability-${g2_array}/manifest.json"
g2_smoke="$(sbatch --parsable --dependency="afterok:${g2_merge}" \
  --export="ALL,GRAF_GRAPH_CACHE_MANIFEST=${graph_manifest},GRAF_VIABILITY_MANIFEST=${viability_manifest},CACHE_BUILD_JOB_ID=${CACHE_BUILD_JOB_ID},G2_VIABILITY_ARRAY_JOB_ID=${g2_array}" \
  reproductions/06_graf_opsd/g2-smoke.sbatch)"
echo "g2_smoke_job=${g2_smoke}"
sbatch --parsable --dependency="afterok:${g2_smoke}" \
  --export="ALL,CACHE_BUILD_JOB_ID=${CACHE_BUILD_JOB_ID},G2_VIABILITY_ARRAY_JOB_ID=${g2_array},G2_SMOKE_JOB_ID=${g2_smoke}" \
  reproductions/06_graf_opsd/after-g2-smoke.sbatch
