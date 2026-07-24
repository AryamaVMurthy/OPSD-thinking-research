#!/usr/bin/env bash
#SBATCH --job-name=opsd-eval
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node10
#SBATCH --gres=gpu:8
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module load u22/cuda/12.4
source "${HOME}/OPSD-thinking-research/infra/turing/common.sh"

: "${CONFIG:?set CONFIG to a repository-relative YAML path}"
: "${EVAL_MODULE:?set EVAL_MODULE to math_eval or lcb_eval}"
: "${RUN_NAME:?set RUN_NAME}"

OUTPUT_DIR="${RESULTS_ROOT}/${RUN_NAME}"
export OUTPUT_DIR
mkdir -p "${OUTPUT_DIR}"
cp "${PROJECT_SOURCE}/${CONFIG}" "${OUTPUT_DIR}/config.yaml"
record_run_manifest "${OUTPUT_DIR}"
python3 -m opsd_research.cli validate-config "${PROJECT_SOURCE}/${CONFIG}"

METHOD="${METHOD:-untouched}"
CHECKPOINT="${CHECKPOINT:-none}"
export CONFIG EVAL_MODULE RUN_NAME ADAPTER METHOD CHECKPOINT

start_gpu_telemetry "${RUN_NAME}"
trap stop_gpu_telemetry EXIT

srun \
  --kill-on-bad-exit=1 \
  --gpus-per-task=1 \
  --gpu-bind=single:1 \
  "${PROJECT_SOURCE}/infra/turing/eval_worker.sh"

if [[ "${EVAL_MODULE}" == "math_eval" ]]; then
  python3 -m opsd_research.summarize_math \
    --config "${PROJECT_SOURCE}/${CONFIG}" \
    --input "${OUTPUT_DIR}"/generations.shard*.jsonl \
    --output "${OUTPUT_DIR}/summary.json"
fi

du -sh "${OUTPUT_DIR}" "${HF_HOME}" "${TRAIN_ROOT}"
