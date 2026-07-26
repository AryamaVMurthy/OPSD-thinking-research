#!/usr/bin/env bash
#SBATCH --job-name=graf-after-g7-smoke
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
: "${CACHE_BUILD_JOB_ID:?}" "${G2_VIABILITY_ARRAY_JOB_ID:?}" "${G7_SMOKE_JOB_ID:?}"
run_name="graf-g7-information-weighted-5-${G7_SMOKE_JOB_ID}"; training_dir="${TRAIN_ROOT}/${run_name}"; log="${PROJECT_SOURCE}/logs/graf-routed-${G7_SMOKE_JOB_ID}.out"
python3 - "${log}" "${training_dir}/training-summary.json" <<'PY'
import json, math, sys
a=[]
for line in open(sys.argv[1],encoding="utf-8",errors="replace"):
 if '"event":"graf_branch_loss"' in line:a.append(float(json.loads(line[line.index("{"):])["active_forks"]))
if not a or max(a)<=0:raise SystemExit("G7 smoke never activated an information-weighted fork")
p=json.load(open(sys.argv[2]));i=p["rollout_dump_integrity"]
if not i["covers_latest_checkpoint"] or i["latest_checkpoint_step"]!=5:raise SystemExit("G7 smoke rollout missing")
if any(not math.isfinite(float(x["loss"])) or not math.isfinite(float(x["grad_norm"])) for x in p["loss_history"]):raise SystemExit("G7 smoke is non-finite")
PY
graph_manifest="${SCRATCH_ROOT}/graf_cache/g1-${CACHE_BUILD_JOB_ID}/manifest.json"; viability_manifest="${SCRATCH_ROOT}/graf_cache/g2-viability-${G2_VIABILITY_ARRAY_JOB_ID}/manifest.json"
pilot="$(GRAF_GRAPH_CACHE_MANIFEST="${graph_manifest}" GRAF_VIABILITY_MANIFEST="${viability_manifest}" GRAF_CANDIDATE_ID=g7-information-weighted GRAF_CANDIDATE_CONFIG=reproductions/06_graf_opsd/configs/g7-information-weighted.yaml GRAF_MAX_STEPS=50 GRAF_PORT=12981 sbatch --parsable reproductions/06_graf_opsd/run-routed-candidate.sbatch)"
echo "g7_pilot_job=${pilot}"
sbatch --parsable --dependency="afterok:${pilot}" --export="ALL,CACHE_BUILD_JOB_ID=${CACHE_BUILD_JOB_ID},G2_VIABILITY_ARRAY_JOB_ID=${G2_VIABILITY_ARRAY_JOB_ID},G7_PILOT_JOB_ID=${pilot}" reproductions/06_graf_opsd/after-g7-pilot.sbatch
