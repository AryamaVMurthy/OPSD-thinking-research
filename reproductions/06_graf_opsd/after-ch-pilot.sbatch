#!/usr/bin/env bash
# Validate the 25-step CH pilot and submit its bounded paired screen.
#SBATCH --job-name=after-ch-pilot
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
: "${CH_PILOT_JOB_ID:?}"
run_name="ch-ch0-contrastive-hindsight-2048-25-${CH_PILOT_JOB_ID}"
training_dir="${TRAIN_ROOT}/${run_name}"
summary="${training_dir}/training-summary.json"
test -f "${summary}"
test -f "${training_dir}/adapter_model.safetensors"
python3 - "${summary}" <<'PY'
import json, math, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
i = p["rollout_dump_integrity"]
if not i["covers_latest_checkpoint"] or i["latest_checkpoint_step"] != 25:
    raise SystemExit("CH pilot lacks its step-25 rollout")
if any(not math.isfinite(float(row["loss"])) or
       not math.isfinite(float(row["grad_norm"])) for row in p["loss_history"]):
    raise SystemExit("CH pilot optimization is non-finite")
PY

eval_run="ch0-contrastive-hindsight-2048-25-aime24-full-context-screen-${CH_PILOT_JOB_ID}"
eval_job="$(
  CONFIG=reproductions/06_graf_opsd/configs/qwen3-4b-aime24-full-context-screen.yaml \
  EVAL_MODULE=math_eval \
  RUN_NAME="${eval_run}" \
  METHOD=ch0-contrastive-hindsight-2048 \
  CHECKPOINT=step-25 \
  ADAPTER="${training_dir}" \
  sbatch --parsable infra/turing/run_eval.sbatch
)"
echo "ch_screen_eval_job=${eval_job}"
sbatch \
  --parsable \
  --dependency="afterok:${eval_job}" \
  --export="ALL,CH_PILOT_JOB_ID=${CH_PILOT_JOB_ID},CH_EVAL_RUN=${eval_run}" \
  reproductions/06_graf_opsd/after-ch-eval.sbatch
