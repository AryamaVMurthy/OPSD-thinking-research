#!/usr/bin/env bash
# Validate the 50-step confirmation, then launch its paired full-context screen.
#SBATCH --job-name=after-ch-diversity-train
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
: "${CH_CONFIRM_TRAIN_JOB_ID:?}" "${CH_CONFIRM_RUN_NAME:?}"

candidate="ch1-completion-aware-4096"
run_name="${CH_CONFIRM_RUN_NAME}"
training_dir="${TRAIN_ROOT}/${run_name}"
summary="${training_dir}/training-summary.json"
test -f "${training_dir}/adapter_model.safetensors"
test -f "${summary}"
python3 - "${summary}" <<'PY'
import json, math, sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
losses = summary["loss_history"]
integrity = summary["rollout_dump_integrity"]
if len(losses) != 50 or int(summary["latest_logged_step"]) != 50:
    raise SystemExit("confirmation training does not contain 50 optimizer steps")
if any(
    not math.isfinite(float(row["loss"]))
    or not math.isfinite(float(row["grad_norm"]))
    for row in losses
):
    raise SystemExit("confirmation optimization contains non-finite values")
if not integrity["covers_latest_checkpoint"] or int(
    integrity["latest_checkpoint_step"]
) != 50:
    raise SystemExit("confirmation rollout dump does not cover checkpoint 50")
print(json.dumps({
    "event": "ch1_diversity_training_gate",
    "optimizer_steps": len(losses),
    "checkpoint_step": integrity["latest_checkpoint_step"],
}, sort_keys=True))
PY

baseline_run="graf-aime24-base-full-context-screen-20260727"
baseline="${RESULTS_ROOT}/${baseline_run}"
baseline_archive="${PROJECT_SOURCE}/artifacts/cross-node/${baseline_run}.tar.gz"
expected_archive_sha256="ac5a7fe78351de7d25b1fb211e81f4d662ea6049bccb9e8bdf2b93fa771567ac"
actual_archive_sha256="$(sha256sum "${baseline_archive}" | awk '{print $1}')"
if [[ "${actual_archive_sha256}" != "${expected_archive_sha256}" ]]; then
  echo "paired baseline staging archive SHA-256 does not match" >&2
  exit 1
fi
if [[ ! -d "${baseline}" ]]; then
  tar -C "${RESULTS_ROOT}" -xzf "${baseline_archive}"
fi
(
  cd "${baseline}"
  sha256sum -c artifact-manifest-16414.sha256
)

eval_run="${candidate}-50-aime24-full-context-screen-${CH_CONFIRM_TRAIN_JOB_ID}"
eval_job="$(
  CONFIG=reproductions/06_graf_opsd/configs/qwen3-4b-aime24-full-context-screen.yaml \
  EVAL_MODULE=math_eval \
  RUN_NAME="${eval_run}" \
  METHOD="${candidate}" \
  CHECKPOINT=step-50 \
  ADAPTER="${training_dir}" \
  sbatch --parsable \
    --nodelist=node10 \
    --gres=gpu:4 \
    --ntasks=4 \
    --cpus-per-task=8 \
    --mem=450G \
    --output="${SCRATCH_ROOT}/slurm-opsd-eval-%j.out" \
    infra/turing/run_eval.sbatch
)"
echo "ch1_diversity_eval_job=${eval_job}"
comparison="$(
  sbatch --parsable \
    --dependency="afterok:${eval_job}" \
    --nodelist=node10 \
    --output="${SCRATCH_ROOT}/slurm-after-ch-diversity-eval-%j.out" \
    --export="ALL,CH_CONFIRM_EVAL_RUN=${eval_run}" \
    reproductions/06_graf_opsd/after-ch-diversity-eval.sbatch
)"
echo "ch1_diversity_eval_controller=${comparison}"
