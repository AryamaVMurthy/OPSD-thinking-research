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
  adapter_args=(--adapter "${ADAPTER}")
fi

python3 -m "opsd_research.${EVAL_MODULE}" \
  --config "${PROJECT_SOURCE}/${CONFIG}" \
  --output "${OUTPUT_DIR}/generations.shard${SLURM_PROCID}.jsonl" \
  --method "${METHOD}" \
  --checkpoint "${CHECKPOINT}" \
  --shard-id "${SLURM_PROCID}" \
  --num-shards "${SLURM_NTASKS}" \
  "${adapter_args[@]}"
