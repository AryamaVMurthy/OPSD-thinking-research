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
e={"event":"g7_development_result","candidate_id":"g7-information-weighted","avg_at_12_delta":d,"delta_ci_lower":lo,"delta_ci_upper":hi,"promotion_gate_passed":d>=.03 and lo>0 and hi>0,"comparison_protocol":c["comparison_protocol"]};e["ledger_event_sha256"]=append_ledger(Path(sys.argv[2]),e);print(json.dumps(e,sort_keys=True))
PY
promote="$(python3 - "${treatment}/paired-vs-base.json" <<'PY'
import json,sys
c=json.load(open(sys.argv[1]));d=c["avg_at_12"]["delta"];lo,hi=c["delta_bootstrap_95ci"]["avg_at_12"];print("1" if d>=.03 and lo>0 and hi>0 else "0")
PY
)"
if [[ "${promote}" == 1 ]]; then
 graph_manifest="${SCRATCH_ROOT}/graf_cache/g1-${CACHE_BUILD_JOB_ID}/manifest.json"; viability_manifest="${SCRATCH_ROOT}/graf_cache/g2-viability-${G2_VIABILITY_ARRAY_JOB_ID}/manifest.json"
 full="$(sbatch --parsable --export="ALL,CANDIDATE_ID=g7-information-weighted,CANDIDATE_CONFIG=reproductions/06_graf_opsd/configs/g7-information-weighted.yaml,METHOD=graf-information-weighted,COMPLETION_LENGTH=2048,GRAF_GRAPH_CACHE_MANIFEST=${graph_manifest},GRAF_VIABILITY_MANIFEST=${viability_manifest}" reproductions/06_graf_opsd/promoted-full-train.sbatch)"
 echo "g7_full_confirmation_job=${full}"
 sbatch --parsable --dependency="afterok:${full}" --export="ALL,FULL_JOB_ID=${full},CANDIDATE_ID=g7-information-weighted,METHOD=graf-information-weighted" reproductions/06_graf_opsd/after-promoted-full-train.sbatch
else
 echo "g7 did not clear the development gate; use its paired result to choose the next measured graph/data mutation."
fi
