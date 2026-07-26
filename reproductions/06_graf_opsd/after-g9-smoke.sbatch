#!/usr/bin/env bash
#SBATCH --job-name=graf-after-g9-smoke
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
: "${CACHE_BUILD_JOB_ID:?}" "${G9_VIABILITY_ARRAY_JOB_ID:?}" "${G9_SMOKE_JOB_ID:?}"
run_name="graf-g9-recovery-conditioned-5-${G9_SMOKE_JOB_ID}"; training_dir="${TRAIN_ROOT}/${run_name}"; log="${PROJECT_SOURCE}/logs/graf-routed-${G9_SMOKE_JOB_ID}.out"
python3 - "${log}" "${training_dir}/training-summary.json" <<'PY'
import json, math, sys
active=[]
for line in open(sys.argv[1],encoding="utf-8",errors="replace"):
 if '"event":"graf_branch_loss"' in line: active.append(float(json.loads(line[line.index("{"):])["active_forks"]))
if not active or max(active)<=0: raise SystemExit("G9 smoke never activated a recovery-conditioned fork")
s=json.load(open(sys.argv[2])); i=s["rollout_dump_integrity"]
if not i["covers_latest_checkpoint"] or i["latest_checkpoint_step"]!=5: raise SystemExit("G9 smoke rollout missing")
if any(not math.isfinite(float(x["loss"])) or not math.isfinite(float(x["grad_norm"])) for x in s["loss_history"]): raise SystemExit("G9 smoke is non-finite")
PY
graph_manifest="${SCRATCH_ROOT}/graf_cache/g1-${CACHE_BUILD_JOB_ID}/manifest.json"; viability_manifest="${SCRATCH_ROOT}/graf_cache/g9-recovery-viability-${G9_VIABILITY_ARRAY_JOB_ID}/manifest.json"
pilot="$(GRAF_GRAPH_CACHE_MANIFEST="${graph_manifest}" GRAF_VIABILITY_MANIFEST="${viability_manifest}" GRAF_CANDIDATE_ID=g9-recovery-conditioned GRAF_CANDIDATE_CONFIG=reproductions/06_graf_opsd/configs/g9-recovery-conditioned.yaml GRAF_MAX_STEPS=50 GRAF_PORT=12983 sbatch --parsable reproductions/06_graf_opsd/run-routed-candidate.sbatch)"
echo "g9_pilot_job=${pilot}"
monitor="$(sbatch --parsable --dependency="after:${pilot}" --export="ALL,MONITOR_JOB_ID=${pilot},MONITOR_LOG=${PROJECT_SOURCE}/logs/graf-routed-${pilot}.out,MONITOR_MAX_COMPLETION_LENGTH=4096,MONITOR_REPORT_STEM=${PROJECT_SOURCE}/logs/graf-g9-pilot-${pilot}.live" reproductions/06_graf_opsd/monitor-live-training.sbatch)"
echo "g9_pilot_monitor_job=${monitor}"
sbatch --parsable --dependency="afterok:${pilot}" --export="ALL,CACHE_BUILD_JOB_ID=${CACHE_BUILD_JOB_ID},G9_VIABILITY_ARRAY_JOB_ID=${G9_VIABILITY_ARRAY_JOB_ID},G9_PILOT_JOB_ID=${pilot}" reproductions/06_graf_opsd/after-g9-pilot.sbatch
