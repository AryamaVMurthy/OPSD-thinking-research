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
python3 -m opsd_research.cli validate-config "${PROJECT_SOURCE}/${CONFIG}"

METHOD="${METHOD:-untouched}"
CHECKPOINT="${CHECKPOINT:-none}"
ADAPTER="${ADAPTER:-}"
ADAPTER_SHA256=""
manifest_extras=()
if [[ -n "${ADAPTER}" ]]; then
  adapter_file="${ADAPTER}/adapter_model.safetensors"
  if [[ ! -f "${adapter_file}" ]]; then
    echo "Missing LoRA adapter: ${adapter_file}" >&2
    exit 1
  fi
  ADAPTER_SHA256="$(sha256sum "${adapter_file}" | awk '{print $1}')"
  {
    printf 'adapter_path=%s\n' "${ADAPTER}"
    printf 'adapter_sha256=%s\n' "${ADAPTER_SHA256}"
    printf 'method=%s\n' "${METHOD}"
    printf 'checkpoint=%s\n' "${CHECKPOINT}"
  } > "${OUTPUT_DIR}/adapter-metadata.txt"
  manifest_extras+=(adapter-metadata.txt)
fi
record_run_manifest "${OUTPUT_DIR}" "${manifest_extras[@]}"
export CONFIG EVAL_MODULE RUN_NAME ADAPTER ADAPTER_SHA256 METHOD CHECKPOINT

start_gpu_telemetry "${RUN_NAME}"
trap stop_gpu_telemetry EXIT

srun \
  --kill-on-bad-exit=1 \
  --gpus-per-task=1 \
  --gpu-bind=single:1 \
  "${PROJECT_SOURCE}/infra/turing/eval_worker.sh"

if [[ "${EVAL_MODULE}" == "math_eval" ]]; then
  score_python="${SCORE_ENV_DIR}/bin/python"
  if [[ ! -x "${score_python}" ]]; then
    score_python="${ENV_DIR}/bin/python"
    echo "{\"event\":\"score_env_fallback\",\"python\":\"${score_python}\"}" >&2
  fi
  "${score_python}" -m opsd_research.rescore_matharena \
    --config "${PROJECT_SOURCE}/${CONFIG}" \
    --input "${OUTPUT_DIR}"/generations.shard*.jsonl \
    --grades-output "${OUTPUT_DIR}/official-grades.jsonl" \
    --summary-output "${OUTPUT_DIR}/summary.json"
fi

"${PROJECT_SOURCE}/infra/turing/artifact_manifest.sh" \
  "${OUTPUT_DIR}" "artifact-manifest-${SLURM_JOB_ID}.sha256"
du -sh "${OUTPUT_DIR}" "${HF_HOME}" "${TRAIN_ROOT}"
