#!/usr/bin/env bash
# C2 establishes the held-out-data control before G4 changes the objective.
#SBATCH --job-name=graf-after-c2-eval
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
: "${C2_PILOT_JOB_ID:?}" "${C2_EVAL_RUN:?}" "${CACHE_BUILD_JOB_ID:?}" "${G2_VIABILITY_ARRAY_JOB_ID:?}"
baseline="${RESULTS_ROOT}/graf-aime24-base-20260726"; treatment="${RESULTS_ROOT}/${C2_EVAL_RUN}"
test -f "${baseline}/summary.json"; test -f "${treatment}/summary.json"
python3 -m opsd_research.compare_math --baseline-input "${baseline}"/generations.shard*.jsonl --baseline-grades "${baseline}/official-grades.jsonl" --treatment-input "${treatment}"/generations.shard*.jsonl --treatment-grades "${treatment}/official-grades.jsonl" --samples-per-problem 12 --output "${treatment}/paired-vs-base.json"
python3 - "${treatment}/paired-vs-base.json" artifacts/graf_opsd/candidate-ledger.jsonl <<'PY'
import json, sys
from pathlib import Path
from opsd_research.graf_contract import append_ledger
c = json.load(open(sys.argv[1], encoding="utf-8")); d = c["avg_at_12"]["delta"]; lo, hi = c["delta_bootstrap_95ci"]["avg_at_12"]
e = {"event":"c2_heldout_control_result","candidate_id":"c2-long-rollout-4096-heldout","avg_at_12_delta":d,"delta_ci_lower":lo,"delta_ci_upper":hi,"comparison_protocol":c["comparison_protocol"]}
e["ledger_event_sha256"] = append_ledger(Path(sys.argv[2]), e); print(json.dumps(e, sort_keys=True))
PY
python3 -m opsd_research.graf_autoresearch register --policy reproductions/06_graf_opsd/configs/autoresearch.yaml --config reproductions/06_graf_opsd/configs/g4-viability-long-4096.yaml --ledger artifacts/graf_opsd/candidate-ledger.jsonl --candidate-id g4-viability-long-4096 --parent-id c2-long-rollout-4096-heldout --stage G4 --hypothesis "Add dynamic viability routing to the held-out-data 4,096-token control, preserving the model, corpus partition, rollout budget, and benchmark protocol." --changed-field graph_mode
graph_manifest="${SCRATCH_ROOT}/graf_cache/g1-${CACHE_BUILD_JOB_ID}/manifest.json"
viability_manifest="${SCRATCH_ROOT}/graf_cache/g2-viability-${G2_VIABILITY_ARRAY_JOB_ID}/manifest.json"
g4_smoke="$(GRAF_GRAPH_CACHE_MANIFEST="${graph_manifest}" GRAF_VIABILITY_MANIFEST="${viability_manifest}" sbatch --parsable reproductions/06_graf_opsd/g4-smoke.sbatch)"
echo "g4_smoke_job=${g4_smoke}"
sbatch --parsable --dependency="afterok:${g4_smoke}" --export="ALL,CACHE_BUILD_JOB_ID=${CACHE_BUILD_JOB_ID},G2_VIABILITY_ARRAY_JOB_ID=${G2_VIABILITY_ARRAY_JOB_ID},G4_SMOKE_JOB_ID=${g4_smoke}" reproductions/06_graf_opsd/after-g4-smoke.sbatch
