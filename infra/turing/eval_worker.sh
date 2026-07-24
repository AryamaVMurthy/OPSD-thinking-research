#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_SOURCE:?}"
: "${CONFIG:?}"
: "${EVAL_MODULE:?}"
: "${OUTPUT_DIR:?}"
: "${METHOD:?}"
: "${CHECKPOINT:?}"
: "${SLURM_PROCID:?}"
: "${SLURM_NTASKS:?}"

source "${ENV_DIR}/bin/activate"
cd "${PROJECT_SOURCE}"

adapter_args=()
if [[ -n "${ADAPTER:-}" ]]; then
  : "${ADAPTER_SHA256:?ADAPTER_SHA256 is required with ADAPTER}"
  adapter_args=(
    --adapter "${ADAPTER}"
    --adapter-sha256 "${ADAPTER_SHA256}"
  )
fi

python3 -m "opsd_research.${EVAL_MODULE}" \
  --config "${PROJECT_SOURCE}/${CONFIG}" \
  --output "${OUTPUT_DIR}/generations.shard${SLURM_PROCID}.jsonl" \
  --method "${METHOD}" \
  --checkpoint "${CHECKPOINT}" \
  --shard-id "${SLURM_PROCID}" \
  --num-shards "${SLURM_NTASKS}" \
  "${adapter_args[@]}"
