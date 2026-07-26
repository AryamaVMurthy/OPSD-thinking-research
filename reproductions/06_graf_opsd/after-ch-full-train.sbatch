#!/usr/bin/env bash
# Validate 200-step CH1 training, then run sequential official AIME24 pairs.
#SBATCH --job-name=after-ch-full-train
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node10
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module load u22/cuda/12.4
source "${HOME}/OPSD-thinking-research/infra/turing/common.sh"
cd "${PROJECT_SOURCE}"
: "${CH_FULL_TRAIN_JOB_ID:?}"

method="ch1-completion-aware-4096-full"
run_name="ch-${method}-200-${CH_FULL_TRAIN_JOB_ID}"
training_dir="${TRAIN_ROOT}/${run_name}"
summary="${training_dir}/training-summary.json"
test -f "${summary}"
test -f "${training_dir}/adapter_model.safetensors"
for step in 50 100 150 200; do
  test -f "${training_dir}/checkpoint-${step}/adapter_model.safetensors"
done
python3 - "${summary}" <<'PY'
import json, math, sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
history = summary["loss_history"]
integrity = summary["rollout_dump_integrity"]
if len(history) != 200:
    raise SystemExit(f"expected 200 loss records, found {len(history)}")
if any(
    not math.isfinite(float(row["loss"]))
    or not math.isfinite(float(row["grad_norm"]))
    for row in history
):
    raise SystemExit("full CH1 optimization contains non-finite values")
if (
    not integrity["covers_latest_checkpoint"]
    or integrity["latest_checkpoint_step"] != 200
):
    raise SystemExit("full CH1 rollout dump does not cover step 200")
print(json.dumps({
    "event": "ch1_full_training_gate",
    "optimizer_steps": len(history),
    "latest_checkpoint_step": integrity["latest_checkpoint_step"],
    "finite": True,
}, sort_keys=True))
PY

treatment_run="${method}-200-aime24-official-${CH_FULL_TRAIN_JOB_ID}"
baseline_run="qwen3-4b-base-aime24-official-${CH_FULL_TRAIN_JOB_ID}"
treatment="$(
  CONFIG=reproductions/06_graf_opsd/configs/qwen3-4b-aime24-dev.yaml \
  EVAL_MODULE=math_eval \
  RUN_NAME="${treatment_run}" \
  METHOD="${method}" \
  CHECKPOINT=step-200 \
  ADAPTER="${training_dir}" \
  sbatch --parsable \
    --nodelist=node10 --gres=gpu:8 --ntasks=8 --cpus-per-task=8 --mem=900G \
    --output="${SCRATCH_ROOT}/slurm-opsd-eval-%j.out" \
    infra/turing/run_eval.sbatch
)"
baseline="$(
  CONFIG=reproductions/06_graf_opsd/configs/qwen3-4b-aime24-dev.yaml \
  EVAL_MODULE=math_eval \
  RUN_NAME="${baseline_run}" \
  METHOD=untouched \
  CHECKPOINT=none \
  sbatch --parsable \
    --dependency="afterok:${treatment}" \
    --nodelist=node10 --gres=gpu:8 --ntasks=8 --cpus-per-task=8 --mem=900G \
    --output="${SCRATCH_ROOT}/slurm-opsd-eval-%j.out" \
    infra/turing/run_eval.sbatch
)"
comparison="$(
  sbatch --parsable \
    --dependency="afterok:${baseline}" \
    --output="${SCRATCH_ROOT}/slurm-after-ch-full-aime24-%j.out" \
    --export="ALL,CH_FULL_TRAIN_JOB_ID=${CH_FULL_TRAIN_JOB_ID},CH_FULL_TREATMENT_RUN=${treatment_run},CH_FULL_BASELINE_RUN=${baseline_run}" \
    reproductions/06_graf_opsd/after-ch-full-aime24.sbatch
)"
printf 'ch1_full_aime24_treatment=%s run=%s\n' "${treatment}" "${treatment_run}"
printf 'ch1_full_aime24_baseline=%s run=%s\n' "${baseline}" "${baseline_run}"
printf 'ch1_full_aime24_comparison=%s\n' "${comparison}"
