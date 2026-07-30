#!/usr/bin/env bash
# Rebuild FiNOD evidence for a completed run without repeating GPU training.
#SBATCH --job-name=summarize-finod
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node10
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail

: "${PROJECT_SOURCE:?}"
: "${FINOD_RUN_NAME:?}"
: "${FINOD_TRAINING_LOG:?}"
: "${FINOD_MAX_COMPLETION_LENGTH:?}"
: "${FINOD_TRAINING_JOB_ID:?}"

source "${HOME}/OPSD-thinking-research/infra/turing/common.sh"
cd "${PROJECT_SOURCE}"
training_dir="${TRAIN_ROOT}/${FINOD_RUN_NAME}"
telemetry="${TELEMETRY_ROOT}/${FINOD_RUN_NAME}-${FINOD_TRAINING_JOB_ID}.csv"

python3 -m opsd_research.summarize_training \
  --training-dir "${training_dir}" \
  --training-log "${FINOD_TRAINING_LOG}" \
  --telemetry "${telemetry}" \
  --output "${training_dir}/training-summary.json" \
  --max-completion-length "${FINOD_MAX_COMPLETION_LENGTH}"
