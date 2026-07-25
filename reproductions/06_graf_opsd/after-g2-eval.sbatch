#!/usr/bin/env bash
# Record the paired official development result for G2.  No locked benchmark
# may be touched unless the same pre-registered promotion gate succeeds.
#SBATCH --job-name=graf-after-g2-eval
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

: "${G2_PILOT_JOB_ID:?missing G2 pilot job ID}"
: "${G2_EVAL_RUN:?missing G2 evaluation run name}"
baseline="${RESULTS_ROOT}/graf-aime24-base-20260726"
treatment="${RESULTS_ROOT}/${G2_EVAL_RUN}"
test -f "${baseline}/summary.json"
test -f "${treatment}/summary.json"
python3 -m opsd_research.compare_math \
  --baseline-input "${baseline}"/generations.shard*.jsonl \
  --baseline-grades "${baseline}/official-grades.jsonl" \
  --treatment-input "${treatment}"/generations.shard*.jsonl \
  --treatment-grades "${treatment}"/official-grades.jsonl \
  --samples-per-problem 12 --output "${treatment}/paired-vs-base.json"
python3 - "${treatment}/paired-vs-base.json" "artifacts/graf_opsd/candidate-ledger.jsonl" <<'PY'
import json
import sys
from pathlib import Path
from opsd_research.graf_contract import append_ledger

comparison = json.load(open(sys.argv[1], encoding="utf-8"))
delta = comparison["avg_at_12"]["delta"]
lower, upper = comparison["delta_bootstrap_95ci"]["avg_at_12"]
event = {
    "event": "g2_development_result",
    "candidate_id": "g2-viability-routed-2048",
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
  adapter="${TRAIN_ROOT}/graf-g2-viability-50-${G2_PILOT_JOB_ID}"
  locked="$(sbatch --parsable --export="ALL,METHOD=g2-viability-routed,ADAPTER=${adapter}" \
    reproductions/06_graf_opsd/submit-locked-evals.sbatch)"
  echo "g2_promoted_locked_submit_job=${locked}"
else
  # G3 differs from G2 in exactly one pre-registered field: the one-sided
  # branch entropy floor. Reuse G2's immutable graph/viability inputs.
  python3 -m opsd_research.graf_autoresearch register \
    --policy reproductions/06_graf_opsd/configs/autoresearch.yaml \
    --config reproductions/06_graf_opsd/configs/g3-viability-entropy.yaml \
    --ledger artifacts/graf_opsd/candidate-ledger.jsonl \
    --candidate-id g3-viability-entropy-2048 \
    --parent-id g2-viability-routed-2048 --stage G3 \
    --hypothesis "Match the full viable-branch entropy floor without rewarding uncertainty beyond the frozen branch target." \
    --changed-field entropy_floor_weight
  graph_manifest="${SCRATCH_ROOT}/graf_cache/g1-${CACHE_BUILD_JOB_ID}/manifest.json"
  viability_manifest="${SCRATCH_ROOT}/graf_cache/g2-viability-${G2_VIABILITY_ARRAY_JOB_ID}/manifest.json"
  g3_smoke="$(sbatch --parsable \
    --export="ALL,GRAF_GRAPH_CACHE_MANIFEST=${graph_manifest},GRAF_VIABILITY_MANIFEST=${viability_manifest}" \
    reproductions/06_graf_opsd/g3-smoke.sbatch)"
  echo "g3_smoke_job=${g3_smoke}"
  sbatch --parsable --dependency="afterok:${g3_smoke}" \
    --export="ALL,CACHE_BUILD_JOB_ID=${CACHE_BUILD_JOB_ID},G2_VIABILITY_ARRAY_JOB_ID=${G2_VIABILITY_ARRAY_JOB_ID},G3_SMOKE_JOB_ID=${g3_smoke}" \
    reproductions/06_graf_opsd/after-g3-smoke.sbatch
fi
