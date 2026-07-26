#!/usr/bin/env bash
#SBATCH --job-name=graf-after-g10-pilot
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
: "${G10_CACHE_BUILD_JOB_ID:?}" "${G10_VIABILITY_ARRAY_JOB_ID:?}" "${G10_PILOT_JOB_ID:?}"
run_name="graf-g10-teacher-critique-50-${G10_PILOT_JOB_ID}"
training_dir="${TRAIN_ROOT}/${run_name}"
log="${PROJECT_SOURCE}/logs/graf-routed-${G10_PILOT_JOB_ID}.out"
python3 - "${log}" "${training_dir}/training-summary.json" <<'PY'
import json, math, sys
active=[]
for line in open(sys.argv[1],encoding="utf-8",errors="replace"):
 if '"event":"graf_branch_loss"' in line: active.append(float(json.loads(line[line.index("{"):])["active_forks"]))
if not active or max(active)<=0: raise SystemExit("G10 pilot never activated a teacher-reviewed branch target")
s=json.load(open(sys.argv[2])); i=s["rollout_dump_integrity"]
if not i["covers_latest_checkpoint"] or i["latest_checkpoint_step"]!=50: raise SystemExit("G10 pilot rollout missing")
if any(not math.isfinite(float(x["loss"])) or not math.isfinite(float(x["grad_norm"])) for x in s["loss_history"]): raise SystemExit("G10 pilot is non-finite")
PY
eval_run="graf-g10-teacher-critique-50-aime24-${G10_PILOT_JOB_ID}"
eval_job="$(CONFIG=reproductions/06_graf_opsd/configs/qwen3-4b-aime24-dev.yaml EVAL_MODULE=math_eval RUN_NAME="${eval_run}" METHOD=graf-teacher-critique CHECKPOINT=step-50 ADAPTER="${training_dir}" sbatch --parsable infra/turing/run_eval.sbatch)"
echo "g10_aime24_eval_job=${eval_job}"
sbatch --parsable --dependency="afterok:${eval_job}" --export="ALL,G10_CACHE_BUILD_JOB_ID=${G10_CACHE_BUILD_JOB_ID},G10_VIABILITY_ARRAY_JOB_ID=${G10_VIABILITY_ARRAY_JOB_ID},G10_EVAL_RUN=${eval_run}" reproductions/06_graf_opsd/after-g10-eval.sbatch
