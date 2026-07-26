#!/usr/bin/env bash
# G4's paired AIME24 decision.  A pass must still repeat from base for 200 steps.
#SBATCH --job-name=graf-after-g4-eval
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
: "${CACHE_BUILD_JOB_ID:?}" "${G2_VIABILITY_ARRAY_JOB_ID:?}" "${G4_EVAL_RUN:?}"
baseline="${RESULTS_ROOT}/graf-aime24-base-20260726"; treatment="${RESULTS_ROOT}/${G4_EVAL_RUN}"
test -f "${baseline}/summary.json"; test -f "${treatment}/summary.json"
python3 -m opsd_research.compare_math --baseline-input "${baseline}"/generations.shard*.jsonl --baseline-grades "${baseline}/official-grades.jsonl" --treatment-input "${treatment}"/generations.shard*.jsonl --treatment-grades "${treatment}/official-grades.jsonl" --samples-per-problem 12 --output "${treatment}/paired-vs-base.json"
python3 - "${treatment}/paired-vs-base.json" artifacts/graf_opsd/candidate-ledger.jsonl <<'PY'
import json, sys
from pathlib import Path
from opsd_research.graf_contract import append_ledger
c=json.load(open(sys.argv[1])); d=c["avg_at_12"]["delta"]; lo,hi=c["delta_bootstrap_95ci"]["avg_at_12"]
e={"event":"g4_development_result","candidate_id":"g4-viability-long-4096","avg_at_12_delta":d,"delta_ci_lower":lo,"delta_ci_upper":hi,"promotion_gate_passed":d>=.03 and lo>0 and hi>0,"comparison_protocol":c["comparison_protocol"]}
e["ledger_event_sha256"]=append_ledger(Path(sys.argv[2]),e); print(json.dumps(e,sort_keys=True))
PY
promote="$(python3 - "${treatment}/paired-vs-base.json" <<'PY'
import json,sys
c=json.load(open(sys.argv[1])); d=c["avg_at_12"]["delta"]; lo,hi=c["delta_bootstrap_95ci"]["avg_at_12"]; print("1" if d>=.03 and lo>0 and hi>0 else "0")
PY
)"
if [[ "${promote}" == 1 ]]; then
  graph_manifest="${SCRATCH_ROOT}/graf_cache/g1-${CACHE_BUILD_JOB_ID}/manifest.json"; viability_manifest="${SCRATCH_ROOT}/graf_cache/g2-viability-${G2_VIABILITY_ARRAY_JOB_ID}/manifest.json"
  full="$(sbatch --parsable --export="ALL,CANDIDATE_ID=g4-viability-long-4096,CANDIDATE_CONFIG=reproductions/06_graf_opsd/configs/g4-viability-long-4096.yaml,METHOD=graf-viability-long-4096,COMPLETION_LENGTH=4096,VLLM_GPU_UTILIZATION=0.35,GRAF_GRAPH_CACHE_MANIFEST=${graph_manifest},GRAF_VIABILITY_MANIFEST=${viability_manifest}" reproductions/06_graf_opsd/promoted-full-train.sbatch)"
  echo "g4_full_confirmation_job=${full}"
  sbatch --parsable --dependency="afterok:${full}" --export="ALL,FULL_JOB_ID=${full},CANDIDATE_ID=g4-viability-long-4096,METHOD=graf-viability-long-4096" reproductions/06_graf_opsd/after-promoted-full-train.sbatch
else
  python3 -m opsd_research.graf_autoresearch register \
    --policy reproductions/06_graf_opsd/configs/autoresearch.yaml \
    --config reproductions/06_graf_opsd/configs/g6-information-routed.yaml \
    --ledger artifacts/graf_opsd/candidate-ledger.jsonl \
    --candidate-id g6-information-routed --parent-id g4-viability-long-4096 --stage G6 \
    --hypothesis "Route every action distribution with adaptive information content, retaining multiple viable and invalid alternatives without a fixed action-count or preference-margin rule." \
    --changed-field fork_information_quantile
  graph_manifest="${SCRATCH_ROOT}/graf_cache/g1-${CACHE_BUILD_JOB_ID}/manifest.json"
  viability_manifest="${SCRATCH_ROOT}/graf_cache/g2-viability-${G2_VIABILITY_ARRAY_JOB_ID}/manifest.json"
  g6_smoke="$(GRAF_GRAPH_CACHE_MANIFEST="${graph_manifest}" GRAF_VIABILITY_MANIFEST="${viability_manifest}" GRAF_CANDIDATE_ID=g6-information-routed GRAF_CANDIDATE_CONFIG=reproductions/06_graf_opsd/configs/g6-information-routed.yaml GRAF_MAX_STEPS=5 GRAF_PORT=12980 sbatch --parsable reproductions/06_graf_opsd/run-routed-candidate.sbatch)"
  echo "g6_smoke_job=${g6_smoke}"
  sbatch --parsable --dependency="afterok:${g6_smoke}" --export="ALL,CACHE_BUILD_JOB_ID=${CACHE_BUILD_JOB_ID},G2_VIABILITY_ARRAY_JOB_ID=${G2_VIABILITY_ARRAY_JOB_ID},G6_SMOKE_JOB_ID=${g6_smoke}" reproductions/06_graf_opsd/after-g6-smoke.sbatch
fi
