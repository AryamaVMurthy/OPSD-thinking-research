#!/usr/bin/env bash
# Require a genuine 200-step artifact, then repeat the development evaluation.
#SBATCH --job-name=graf-after-promoted-full
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
: "${FULL_JOB_ID:?}" "${CANDIDATE_ID:?}" "${METHOD:?}"
run_name="graf-${CANDIDATE_ID}-200-${FULL_JOB_ID}"
training_dir="${TRAIN_ROOT}/${run_name}"
test -f "${training_dir}/adapter_model.safetensors"; test -f "${training_dir}/training-summary.json"
python3 - "${training_dir}/training-summary.json" <<'PY'
import json, math, sys
p=json.load(open(sys.argv[1])); i=p["rollout_dump_integrity"]
if not i["covers_latest_checkpoint"] or i["latest_checkpoint_step"] != 200: raise SystemExit("full confirmation lacks a complete step-200 artifact")
if any(not math.isfinite(float(x["loss"])) or not math.isfinite(float(x["grad_norm"])) for x in p["loss_history"]): raise SystemExit("non-finite full confirmation")
PY
eval_run="graf-${CANDIDATE_ID}-200-aime24-${FULL_JOB_ID}"
eval_job="$(CONFIG=reproductions/06_graf_opsd/configs/qwen3-4b-aime24-dev.yaml EVAL_MODULE=math_eval RUN_NAME="${eval_run}" METHOD="${METHOD}-200" CHECKPOINT=step-200 ADAPTER="${training_dir}" sbatch --parsable infra/turing/run_eval.sbatch)"
echo "full_aime24_eval_job=${eval_job}"
sbatch --parsable --dependency="afterok:${eval_job}" --export="ALL,CANDIDATE_ID=${CANDIDATE_ID},METHOD=${METHOD},FULL_JOB_ID=${FULL_JOB_ID},FULL_EVAL_RUN=${eval_run}" reproductions/06_graf_opsd/after-promoted-full-eval.sbatch
