#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_SOURCE:?}" "${ENV_DIR:?}" "${CH_RUN_DIR:?}"
: "${CH_LIMIT:?}" "${CH_BLIND_ATTEMPTS:?}" "${CH_AUDIT_ATTEMPTS:?}"
: "${CH_BLIND_MAX_TOKENS:?}" "${CH_AUDIT_MAX_TOKENS:?}" "${CH_MAX_MODEL_LEN:?}"
: "${SLURM_JOB_ID:?}" "${SLURM_PROCID:?}" "${SLURM_NTASKS:?}"

source "${PROJECT_SOURCE}/infra/turing/vllm_port.sh"
export VLLM_PORT="$(
  vllm_port_for_task "${SLURM_JOB_ID}" "${SLURM_PROCID}"
)"
printf '{"event":"ch_vllm_port_assignment","job_id":"%s","shard_id":%s,"port":%s}\n' \
  "${SLURM_JOB_ID}" "${SLURM_PROCID}" "${VLLM_PORT}"

source "${ENV_DIR}/bin/activate"
cd "${PROJECT_SOURCE}"
python3 -m opsd_research.build_ch_cache_shard \
  --output "${CH_RUN_DIR}/dossiers.shard${SLURM_PROCID}.jsonl" \
  --limit "${CH_LIMIT}" \
  --blind-attempts "${CH_BLIND_ATTEMPTS}" \
  --audit-attempts "${CH_AUDIT_ATTEMPTS}" \
  --blind-max-tokens "${CH_BLIND_MAX_TOKENS}" \
  --audit-max-tokens "${CH_AUDIT_MAX_TOKENS}" \
  --max-model-len "${CH_MAX_MODEL_LEN}" \
  --shard-id "${SLURM_PROCID}" \
  --num-shards "${SLURM_NTASKS}" \
  --seed 42
