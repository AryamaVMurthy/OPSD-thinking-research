#!/usr/bin/env bash
# Validate C2 and submit its paired official AIME24 evaluation.
#SBATCH --job-name=graf-after-c2-pilot
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
: "${C2_PILOT_JOB_ID:?missing C2 pilot job}"
: "${CACHE_BUILD_JOB_ID:?missing graph cache job}" "${G2_VIABILITY_ARRAY_JOB_ID:?missing viability cache job}"
run_name="graf-c2-long-4096-heldout-50-${C2_PILOT_JOB_ID}"
training_dir="${TRAIN_ROOT}/${run_name}"
summary="${training_dir}/training-summary.json"
test -f "${summary}"
test -f "${training_dir}/adapter_model.safetensors"
python3 - "${summary}" <<'PY'
import json, math, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
integrity = payload["rollout_dump_integrity"]
if not integrity["covers_latest_checkpoint"] or integrity["latest_checkpoint_step"] != 50:
    raise SystemExit("C2 pilot lacks a complete final checkpoint/rollout dump")
for row in payload["loss_history"]:
    if not math.isfinite(float(row["loss"])) or not math.isfinite(float(row["grad_norm"])):
        raise SystemExit("C2 pilot loss or gradient was non-finite")
PY
eval_run="graf-c2-long-4096-heldout-50-aime24-${C2_PILOT_JOB_ID}"
eval_job="$(CONFIG=reproductions/06_graf_opsd/configs/qwen3-4b-aime24-dev.yaml EVAL_MODULE=math_eval RUN_NAME="${eval_run}" METHOD=opsd-long-4096-heldout CHECKPOINT=step-50 ADAPTER="${training_dir}" sbatch --parsable infra/turing/run_eval.sbatch)"
echo "c2_aime24_eval_job=${eval_job}"
sbatch --parsable --dependency="afterok:${eval_job}" \
  --export="ALL,C2_PILOT_JOB_ID=${C2_PILOT_JOB_ID},C2_EVAL_RUN=${eval_run},CACHE_BUILD_JOB_ID=${CACHE_BUILD_JOB_ID},G2_VIABILITY_ARRAY_JOB_ID=${G2_VIABILITY_ARRAY_JOB_ID}" \
  reproductions/06_graf_opsd/after-c2-eval.sbatch
