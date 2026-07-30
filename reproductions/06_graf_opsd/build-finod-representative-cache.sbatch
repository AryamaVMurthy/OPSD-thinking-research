#!/usr/bin/env bash
# Build four disjoint shards of a uniform 2,400-problem source sample.
#SBATCH --job-name=finod-cache
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node03
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=350G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail

module load u22/cuda/12.4
: "${PROJECT_SOURCE:?}"
source "${HOME}/OPSD-thinking-research/infra/turing/common.sh"
cd "${PROJECT_SOURCE}"
test -z "$(git status --short)"

export FINOD_GRAPH_LIMIT="${FINOD_GRAPH_LIMIT:-2400}"
export FINOD_SELECTION_SEED="${FINOD_SELECTION_SEED:-73}"
export FINOD_CACHE_SHARDS=4
export FINOD_CACHE_ROOT="${FINOD_CACHE_ROOT:-${SCRATCH_ROOT}/finod_cache/representative-${SLURM_JOB_ID}}"
mkdir -p "${FINOD_CACHE_ROOT}"
git rev-parse HEAD > "${FINOD_CACHE_ROOT}/source-commit.txt"

start_gpu_telemetry "finod-cache-${SLURM_JOB_ID}"
trap stop_gpu_telemetry EXIT
srun \
  --kill-on-bad-exit=1 \
  --gpus-per-task=1 \
  --gpu-bind=single:1 \
  "${PROJECT_SOURCE}/infra/turing/build_finod_cache_worker.sh"

manifests=()
for shard in 0 1 2 3; do
  manifests+=("${FINOD_CACHE_ROOT}/shard-${shard}/manifest.json")
done
python3 -m opsd_research.merge_graf_cache \
  --manifest "${manifests[@]}" \
  --output "${FINOD_CACHE_ROOT}/graphs.jsonl" \
  --manifest-output "${FINOD_CACHE_ROOT}/manifest.json"
python3 -m opsd_research.audit_finod_cache \
  --manifest "${FINOD_CACHE_ROOT}/manifest.json" \
  --output "${FINOD_CACHE_ROOT}/audit.json"
sha256sum \
  "${FINOD_CACHE_ROOT}/graphs.jsonl" \
  "${FINOD_CACHE_ROOT}/manifest.json" \
  "${FINOD_CACHE_ROOT}/audit.json" \
  "${FINOD_CACHE_ROOT}/source-commit.txt" \
  > "${FINOD_CACHE_ROOT}/artifact-manifest.sha256"
printf 'GRAF_GRAPH_CACHE_MANIFEST=%s\n' \
  "${FINOD_CACHE_ROOT}/manifest.json"
