#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_SOURCE:?}"
: "${FINOD_CACHE_ROOT:?}"
: "${FINOD_GRAPH_LIMIT:?}"
: "${FINOD_SELECTION_SEED:?}"
: "${FINOD_CACHE_SHARDS:?}"
: "${SLURM_PROCID:?}"

source "${HOME}/OPSD-thinking-research/infra/turing/common.sh"
cd "${PROJECT_SOURCE}"
shard_dir="${FINOD_CACHE_ROOT}/shard-${SLURM_PROCID}"
mkdir -p "${shard_dir}"

python3 -m opsd_research.build_graf_cache \
  --output "${shard_dir}/graphs.jsonl" \
  --manifest "${shard_dir}/manifest.json" \
  --limit "${FINOD_GRAPH_LIMIT}" \
  --attempts 3 \
  --tensor-parallel-size 1 \
  --answer-blind \
  --seed 42 \
  --selection-seed "${FINOD_SELECTION_SEED}" \
  --shard-id "${SLURM_PROCID}" \
  --num-shards "${FINOD_CACHE_SHARDS}"
