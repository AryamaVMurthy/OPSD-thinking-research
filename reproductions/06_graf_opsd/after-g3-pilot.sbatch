#!/usr/bin/env bash
# Validate a completed G3 pilot, then run the official paired AIME24 protocol.
#SBATCH --job-name=graf-after-g3-pilot
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

: "${G3_PILOT_JOB_ID:?missing G3 pilot job ID}"
run_name="graf-g3-viability-entropy-50-${G3_PILOT_JOB_ID}"
training_dir="${TRAIN_ROOT}/${run_name}"
summary="${training_dir}/training-summary.json"
log="${PROJECT_SOURCE}/logs/graf-g3-pilot-${G3_PILOT_JOB_ID}.out"
adapter="${training_dir}/adapter_model.safetensors"
test -f "${summary}"
test -f "${adapter}"
grep -F '"event":"graf_routed_branch_loss_enabled"' "${log}" >/dev/null
grep -F '"event":"graf_branch_loss"' "${log}" >/dev/null
python3 - "${log}" <<'PY'
import json
import sys

active = []
for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    if '"event":"graf_branch_loss"' in line:
        active.append(float(json.loads(line[line.index("{"):])['active_forks']))
if not active or max(active) <= 0:
    raise SystemExit("G3 pilot never activated a routed branch target")
PY
python3 - "${summary}" <<'PY'
import json, math, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
integrity = payload["rollout_dump_integrity"]
if not integrity["covers_latest_checkpoint"] or integrity["latest_checkpoint_step"] != 50:
    raise SystemExit("G3 pilot lacks a complete final checkpoint/rollout dump")
for row in payload["loss_history"]:
    if not math.isfinite(float(row["loss"])) or not math.isfinite(float(row["grad_norm"])):
        raise SystemExit("G3 pilot loss or gradient was non-finite")
PY
eval_run="graf-g3-viability-entropy-2048-50-aime24-${G3_PILOT_JOB_ID}"
eval_job="$(CONFIG=reproductions/06_graf_opsd/configs/qwen3-4b-aime24-dev.yaml \
  EVAL_MODULE=math_eval RUN_NAME="${eval_run}" METHOD=graf-viability-entropy \
  CHECKPOINT=step-50 ADAPTER="${training_dir}" sbatch --parsable infra/turing/run_eval.sbatch)"
echo "g3_aime24_eval_job=${eval_job}"
sbatch --parsable --dependency="afterok:${eval_job}" \
  --export="ALL,G3_PILOT_JOB_ID=${G3_PILOT_JOB_ID},G3_EVAL_RUN=${eval_run}" \
  reproductions/06_graf_opsd/after-g3-eval.sbatch
