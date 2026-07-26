#!/usr/bin/env bash
#SBATCH --job-name=graf-after-g8-eval
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node03
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
module load u22/cuda/12.4
source "${HOME}/OPSD-thinking-research/infra/turing/common.sh"
cd "${PROJECT_SOURCE}"
: "${CACHE_BUILD_JOB_ID:?}" "${G2_VIABILITY_ARRAY_JOB_ID:?}" "${G8_EVAL_RUN:?}"
treatment="${RESULTS_ROOT}/${G8_EVAL_RUN}"; baseline="${RESULTS_ROOT}/graf-aime24-base-20260726"
python3 -m opsd_research.compare_math --baseline "${baseline}/records.jsonl" --treatment "${treatment}/records.jsonl" --samples-per-problem 12 --output "${treatment}/paired-vs-base.json"
python3 - "${treatment}/paired-vs-base.json" artifacts/graf_opsd/candidate-ledger.jsonl <<'PY'
import json,sys
from pathlib import Path
from opsd_research.graf_contract import append_ledger
c=json.load(open(sys.argv[1])); d=c["avg_at_12"]["delta"]; lo,hi=c["delta_bootstrap_95ci"]["avg_at_12"]
e={"event":"g8_development_result","candidate_id":"g8-posterior-calibrated","avg_at_12_delta":d,"delta_ci_lower":lo,"delta_ci_upper":hi,"selection_metric":"official_aime24_avg_at_12","selection_rule":"delta_gt_zero","comparison_protocol":c["comparison_protocol"]}
e["ledger_event_sha256"]=append_ledger(Path(sys.argv[2]),e); print(json.dumps(e,sort_keys=True))
PY
promote="$(python3 - "${treatment}/paired-vs-base.json" <<'PY'
import json,sys
c=json.load(open(sys.argv[1])); print("1" if c["avg_at_12"]["delta"]>0 else "0")
PY
)"
if [[ "${promote}" == 1 ]]; then
 graph_manifest="${SCRATCH_ROOT}/graf_cache/g1-${CACHE_BUILD_JOB_ID}/manifest.json"; viability_manifest="${SCRATCH_ROOT}/graf_cache/g2-viability-${G2_VIABILITY_ARRAY_JOB_ID}/manifest.json"
 full="$(sbatch --parsable --export="ALL,CANDIDATE_ID=g8-posterior-calibrated,CANDIDATE_CONFIG=reproductions/06_graf_opsd/configs/g8-posterior-calibrated.yaml,METHOD=graf-posterior-calibrated,COMPLETION_LENGTH=4096,VLLM_GPU_UTILIZATION=0.35,GRAF_GRAPH_CACHE_MANIFEST=${graph_manifest},GRAF_VIABILITY_MANIFEST=${viability_manifest}" reproductions/06_graf_opsd/promoted-full-train.sbatch)"
 echo "g8_full_confirmation_job=${full}"
 sbatch --parsable --dependency="afterok:${full}" --export="ALL,FULL_JOB_ID=${full},CANDIDATE_ID=g8-posterior-calibrated,METHOD=graf-posterior-calibrated" reproductions/06_graf_opsd/after-promoted-full-train.sbatch
else
 echo "G8 did not improve AIME24 Avg@12; the ledger and auditable reports preserve it for the next graph/data intervention."
fi
