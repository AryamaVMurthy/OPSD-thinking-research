#!/usr/bin/env bash
#SBATCH --job-name=graf-after-g1-pilot
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

: "${CACHE_BUILD_JOB_ID:?missing graph-cache build job ID}"
: "${G1_PILOT_JOB_ID:?missing G1 pilot job ID}"
run_name="graf-g1-answer-masked-50-${G1_PILOT_JOB_ID}"
training_dir="${TRAIN_ROOT}/${run_name}"
summary="${training_dir}/training-summary.json"
log="${PROJECT_SOURCE}/logs/graf-g1-pilot-${G1_PILOT_JOB_ID}.out"
adapter="${training_dir}/adapter_model.safetensors"
test -f "${summary}"
test -f "${adapter}"
grep -F '"event":"graf_answer_masked_scaffold_enabled"' "${log}" >/dev/null
python3 - "${summary}" <<'PY'
import json
import math
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
integrity = payload["rollout_dump_integrity"]
if not integrity["covers_latest_checkpoint"] or integrity["latest_checkpoint_step"] != 50:
    raise SystemExit("G1 pilot lacks a complete final checkpoint/rollout dump")
for row in payload["training_log_steps"]:
    if not math.isfinite(float(row["loss"])) or not math.isfinite(float(row["grad_norm"])):
        raise SystemExit("G1 pilot loss or gradient was non-finite")
PY

eval_run="graf-g1-scaffold-2048-50-aime24-${G1_PILOT_JOB_ID}"
eval_job="$(CONFIG=reproductions/06_graf_opsd/configs/qwen3-4b-aime24-dev.yaml \
  EVAL_MODULE=math_eval RUN_NAME="${eval_run}" METHOD=graf-answer-masked-scaffold \
  CHECKPOINT=step-50 ADAPTER="${training_dir}" sbatch --parsable infra/turing/run_eval.sbatch)"
echo "g1_aime24_eval_job=${eval_job}"
sbatch --parsable --dependency="afterok:${eval_job}" \
  --export="ALL,G1_PILOT_JOB_ID=${G1_PILOT_JOB_ID},G1_EVAL_RUN=${eval_run}" \
  reproductions/06_graf_opsd/after-g1-eval.sbatch
