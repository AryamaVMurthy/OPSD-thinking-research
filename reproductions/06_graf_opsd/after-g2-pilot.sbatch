#!/usr/bin/env bash
# Validate a completed G2 pilot, then submit the official AIME24 development run.
#SBATCH --job-name=graf-after-g2-pilot
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node10
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module load u22/cuda/12.4
source "${HOME}/OPSD-thinking-research/infra/turing/common.sh"
cd "${PROJECT_SOURCE}"

: "${G2_PILOT_JOB_ID:?missing G2 pilot job ID}"
run_name="graf-g2-viability-50-${G2_PILOT_JOB_ID}"
training_dir="${TRAIN_ROOT}/${run_name}"
summary="${training_dir}/training-summary.json"
log="${PROJECT_SOURCE}/logs/graf-g2-pilot-${G2_PILOT_JOB_ID}.out"
adapter="${training_dir}/adapter_model.safetensors"
test -f "${summary}"
test -f "${adapter}"
grep -F '"event":"graf_routed_branch_loss_enabled"' "${log}" >/dev/null
grep -F '"event":"graf_branch_loss"' "${log}" >/dev/null
python3 - "${summary}" <<'PY'
import json
import math
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
integrity = payload["rollout_dump_integrity"]
if not integrity["covers_latest_checkpoint"] or integrity["latest_checkpoint_step"] != 50:
    raise SystemExit("G2 pilot lacks a complete final checkpoint/rollout dump")
for row in payload["loss_history"]:
    if not math.isfinite(float(row["loss"])) or not math.isfinite(float(row["grad_norm"])):
        raise SystemExit("G2 pilot loss or gradient was non-finite")
PY
eval_run="graf-g2-viability-2048-50-aime24-${G2_PILOT_JOB_ID}"
eval_job="$(CONFIG=reproductions/06_graf_opsd/configs/qwen3-4b-aime24-dev.yaml \
  EVAL_MODULE=math_eval RUN_NAME="${eval_run}" METHOD=graf-viability-routed \
  CHECKPOINT=step-50 ADAPTER="${training_dir}" sbatch --parsable infra/turing/run_eval.sbatch)"
echo "g2_aime24_eval_job=${eval_job}"
sbatch --parsable --dependency="afterok:${eval_job}" \
  --export="ALL,G2_PILOT_JOB_ID=${G2_PILOT_JOB_ID},G2_EVAL_RUN=${eval_run}" \
  reproductions/06_graf_opsd/after-g2-eval.sbatch
