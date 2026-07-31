#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_SOURCE:?}"
: "${FISHER_GUIDANCE_ROOT:?}"
: "${FISHER_GUIDANCE_LIMIT:?}"
: "${FISHER_GUIDANCE_SHARDS:?}"
: "${SLURM_PROCID:?}"

source "${HOME}/OPSD-thinking-research/infra/turing/common.sh"
cd "${PROJECT_SOURCE}"
shard_dir="${FISHER_GUIDANCE_ROOT}/shard-${SLURM_PROCID}"
mkdir -p "${shard_dir}"

python3 -m opsd_research.build_fisher_guidance \
  --output "${shard_dir}/plans.jsonl" \
  --manifest "${shard_dir}/manifest.json" \
  --limit "${FISHER_GUIDANCE_LIMIT}" \
  --plans-per-problem 3 \
  --attempts 4 \
  --seed 731 \
  --selection-seed 73 \
  --data-source amc_aime \
  --data-source aops_forum \
  --shard-id "${SLURM_PROCID}" \
  --num-shards "${FISHER_GUIDANCE_SHARDS}"
