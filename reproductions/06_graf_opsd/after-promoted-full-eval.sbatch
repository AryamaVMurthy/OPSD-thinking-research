#!/usr/bin/env bash
# Only the repeated 200-step paired AIME24 gate may unlock AIME25/AIME26.
#SBATCH --job-name=graf-after-full-eval
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
: "${CANDIDATE_ID:?}" "${METHOD:?}" "${FULL_JOB_ID:?}" "${FULL_EVAL_RUN:?}"
base="${RESULTS_ROOT}/graf-aime24-base-20260726"; treatment="${RESULTS_ROOT}/${FULL_EVAL_RUN}"
python3 -m opsd_research.compare_math --baseline-input "${base}"/generations.shard*.jsonl --baseline-grades "${base}/official-grades.jsonl" --treatment-input "${treatment}"/generations.shard*.jsonl --treatment-grades "${treatment}/official-grades.jsonl" --samples-per-problem 12 --output "${treatment}/paired-vs-base.json"
python3 - "${treatment}/paired-vs-base.json" "artifacts/graf_opsd/candidate-ledger.jsonl" "${CANDIDATE_ID}" <<'PY'
import json,sys
from pathlib import Path
from opsd_research.graf_contract import append_ledger
c=json.load(open(sys.argv[1])); d=c["avg_at_12"]["delta"]; lo,hi=c["delta_bootstrap_95ci"]["avg_at_12"]
e={"event":"full_200_development_result","candidate_id":sys.argv[3],"avg_at_12_delta":d,"delta_ci_lower":lo,"delta_ci_upper":hi,"promotion_gate_passed":d>=.03 and lo>0 and hi>0,"comparison_protocol":c["comparison_protocol"]}; e["ledger_event_sha256"]=append_ledger(Path(sys.argv[2]),e); print(json.dumps(e,sort_keys=True))
PY
promote="$(python3 - "${treatment}/paired-vs-base.json" <<'PY'
import json,sys
c=json.load(open(sys.argv[1])); d=c["avg_at_12"]["delta"]; lo,hi=c["delta_bootstrap_95ci"]["avg_at_12"]; print("1" if d>=.03 and lo>0 and hi>0 else "0")
PY
)"
if [[ "${promote}" == 1 ]]; then adapter="${TRAIN_ROOT}/graf-${CANDIDATE_ID}-200-${FULL_JOB_ID}"; sbatch --parsable --export="ALL,METHOD=${METHOD}-200,ADAPTER=${adapter}" reproductions/06_graf_opsd/submit-locked-evals.sbatch; else echo "full 200-step confirmation did not clear the gate"; fi
