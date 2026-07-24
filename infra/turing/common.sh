#!/usr/bin/env bash
set -euo pipefail

PROJECT_SOURCE="${PROJECT_SOURCE:-${HOME}/OPSD-thinking-research}"
NODE_NAME="${SLURMD_NODENAME:-$(hostname -s)}"
SCRATCH_ROOT="/scratch/${NODE_NAME}/${USER}/opsd-thinking-research"
ENV_DIR="${SCRATCH_ROOT}/env"
HF_HOME="${SCRATCH_ROOT}/hf"
HF_HUB_CACHE="${HF_HOME}/hub"
HF_DATASETS_CACHE="${HF_HOME}/datasets"
RESULTS_ROOT="${SCRATCH_ROOT}/results"
TRAIN_ROOT="${SCRATCH_ROOT}/training"
TELEMETRY_ROOT="${SCRATCH_ROOT}/telemetry"
WANDB_DIR="${SCRATCH_ROOT}/wandb"

export PROJECT_SOURCE SCRATCH_ROOT ENV_DIR
export HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE
export RESULTS_ROOT TRAIN_ROOT TELEMETRY_ROOT WANDB_DIR
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-offline}"

mkdir -p \
  "${SCRATCH_ROOT}" "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" \
  "${RESULTS_ROOT}" "${TRAIN_ROOT}" "${TELEMETRY_ROOT}" "${WANDB_DIR}"

available_kb="$(df -Pk "${SCRATCH_ROOT}" | awk 'NR==2 {print $4}')"
if (( available_kb < 200 * 1024 * 1024 )); then
  echo "Refusing to run with less than 200 GiB free in ${SCRATCH_ROOT}" >&2
  exit 1
fi

if [[ -f "${ENV_DIR}/bin/activate" ]]; then
  source "${ENV_DIR}/bin/activate"
fi

start_gpu_telemetry() {
  local label="$1"
  local path="${TELEMETRY_ROOT}/${label}-${SLURM_JOB_ID:-manual}.csv"
  (
    while true; do
      nvidia-smi \
        --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
        --format=csv,noheader,nounits >> "${path}"
      sleep 30
    done
  ) &
  GPU_TELEMETRY_PID=$!
  export GPU_TELEMETRY_PID
}

stop_gpu_telemetry() {
  if [[ -n "${GPU_TELEMETRY_PID:-}" ]]; then
    kill "${GPU_TELEMETRY_PID}" 2>/dev/null || true
    wait "${GPU_TELEMETRY_PID}" 2>/dev/null || true
  fi
}

record_run_manifest() {
  local output_dir="$1"
  local run_id="${SLURM_JOB_ID:-manual}"
  local runtime_file="runtime-${run_id}.txt"
  local checksum_file="manifest-${run_id}.sha256"
  if ! git -C "${PROJECT_SOURCE}" diff --quiet ||
    ! git -C "${PROJECT_SOURCE}" diff --cached --quiet; then
    echo "Refusing to run from a source tree with tracked modifications" >&2
    return 1
  fi

  git -C "${PROJECT_SOURCE}" rev-parse HEAD > "${output_dir}/source-commit.txt"
  if [[ -f "${SCRATCH_ROOT}/environment-freeze.txt" ]]; then
    cp "${SCRATCH_ROOT}/environment-freeze.txt" "${output_dir}/environment-freeze.txt"
  fi
  {
    printf 'recorded_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'slurm_job_id=%s\n' "${run_id}"
    printf 'node=%s\n' "${NODE_NAME}"
    printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
  } > "${output_dir}/${runtime_file}"
  (
    cd "${output_dir}"
    sha256sum config.yaml source-commit.txt environment-freeze.txt "${runtime_file}" \
      > "${checksum_file}"
  )
}
