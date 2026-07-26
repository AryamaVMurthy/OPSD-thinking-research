#!/usr/bin/env bash
#SBATCH --job-name=graf-after-g7-eval
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
: "${CACHE_BUILD_JOB_ID:?}" "${G2_VIABILITY_ARRAY_JOB_ID:?}" "${G7_EVAL_RUN:?}"
baseline="${RESULTS_ROOT}/graf-aime24-base-20260726"; treatment="${RESULTS_ROOT}/${G7_EVAL_RUN}"
python3 -m opsd_research.compare_math --baseline-input "${baseline}"/generations.shard*.jsonl --baseline-grades "${baseline}/official-grades.jsonl" --treatment-input "${treatment}"/generations.shard*.jsonl --treatment-grades "${treatment}/official-grades.jsonl" --samples-per-problem 12 --output "${treatment}/paired-vs-base.json"
python3 - "${treatment}/paired-vs-base.json" artifacts/graf_opsd/candidate-ledger.jsonl <<'PY'
import json,sys
from pathlib import Path
from opsd_research.graf_contract import append_ledger
c=json.load(open(sys.argv[1]));d=c["avg_at_12"]["delta"];lo,hi=c["delta_bootstrap_95ci"]["avg_at_12"]
e={"event":"g7_development_result","candidate_id":"g7-information-weighted","avg_at_12_delta":d,"delta_ci_lower":lo,"delta_ci_upper":hi,"selection_metric":"official_aime24_avg_at_12","selection_rule":"delta_gt_zero","comparison_protocol":c["comparison_protocol"]};e["ledger_event_sha256"]=append_ledger(Path(sys.argv[2]),e);print(json.dumps(e,sort_keys=True))
PY
promote="$(python3 - "${treatment}/paired-vs-base.json" <<'PY'
import json,sys
c=json.load(open(sys.argv[1]));d=c["avg_at_12"]["delta"];print("1" if d>0 else "0")
PY
)"
if [[ "${promote}" == 1 ]]; then
 graph_manifest="${SCRATCH_ROOT}/graf_cache/g1-${CACHE_BUILD_JOB_ID}/manifest.json"; viability_manifest="${SCRATCH_ROOT}/graf_cache/g2-viability-${G2_VIABILITY_ARRAY_JOB_ID}/manifest.json"
 full="$(sbatch --parsable --export="ALL,CANDIDATE_ID=g7-information-weighted,CANDIDATE_CONFIG=reproductions/06_graf_opsd/configs/g7-information-weighted.yaml,METHOD=graf-information-weighted,COMPLETION_LENGTH=4096,VLLM_GPU_UTILIZATION=0.35,GRAF_GRAPH_CACHE_MANIFEST=${graph_manifest},GRAF_VIABILITY_MANIFEST=${viability_manifest}" reproductions/06_graf_opsd/promoted-full-train.sbatch)"
 echo "g7_full_confirmation_job=${full}"
 sbatch --parsable --dependency="afterok:${full}" --export="ALL,FULL_JOB_ID=${full},CANDIDATE_ID=g7-information-weighted,METHOD=graf-information-weighted" reproductions/06_graf_opsd/after-promoted-full-train.sbatch
else
 python3 -m opsd_research.graf_autoresearch register \
   --policy reproductions/06_graf_opsd/configs/autoresearch.yaml \
   --config reproductions/06_graf_opsd/configs/g8-posterior-calibrated.yaml \
   --ledger artifacts/graf_opsd/candidate-ledger.jsonl \
   --candidate-id g8-posterior-calibrated --parent-id g7-information-weighted --stage G8 \
   --hypothesis "Replace raw finite-sample forced-continuation rates with symmetric-Beta posterior means before dynamic action-distribution distillation, reducing spurious certainty without fixing a graph arity or class pattern." \
   --changed-field viability_beta_prior
 graph_manifest="${SCRATCH_ROOT}/graf_cache/g1-${CACHE_BUILD_JOB_ID}/manifest.json"; viability_manifest="${SCRATCH_ROOT}/graf_cache/g2-viability-${G2_VIABILITY_ARRAY_JOB_ID}/manifest.json"
 g8_smoke="$(GRAF_GRAPH_CACHE_MANIFEST="${graph_manifest}" GRAF_VIABILITY_MANIFEST="${viability_manifest}" GRAF_CANDIDATE_ID=g8-posterior-calibrated GRAF_CANDIDATE_CONFIG=reproductions/06_graf_opsd/configs/g8-posterior-calibrated.yaml GRAF_MAX_STEPS=5 GRAF_PORT=12982 sbatch --parsable reproductions/06_graf_opsd/run-routed-candidate.sbatch)"
 echo "g8_smoke_job=${g8_smoke}"
 sbatch --parsable --dependency="afterok:${g8_smoke}" --export="ALL,CACHE_BUILD_JOB_ID=${CACHE_BUILD_JOB_ID},G2_VIABILITY_ARRAY_JOB_ID=${G2_VIABILITY_ARRAY_JOB_ID},G8_SMOKE_JOB_ID=${g8_smoke}" reproductions/06_graf_opsd/after-g8-smoke.sbatch
fi
