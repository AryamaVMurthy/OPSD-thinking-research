#!/usr/bin/env bash
#SBATCH --job-name=graf-graph-cache
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node03
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module load u22/cuda/12.4
source "${HOME}/OPSD-thinking-research/infra/turing/common.sh"
cd "${PROJECT_SOURCE}"

GRAPH_LIMIT="${GRAPH_LIMIT:-512}"
GRAPH_ATTEMPTS="${GRAPH_ATTEMPTS:-3}"
if (( GRAPH_LIMIT < 1 || GRAPH_ATTEMPTS < 1 )); then
  echo "GRAPH_LIMIT and GRAPH_ATTEMPTS must be positive" >&2
  exit 2
fi
run_dir="${SCRATCH_ROOT}/graf_cache/g1-${SLURM_JOB_ID}"
mkdir -p "${run_dir}"
git rev-parse HEAD > "${run_dir}/source-commit.txt"
start_gpu_telemetry "graf-graph-cache"
trap stop_gpu_telemetry EXIT

python3 -m opsd_research.build_graf_cache \
  --output "${run_dir}/graphs.jsonl" \
  --manifest "${run_dir}/manifest.json" \
  --limit "${GRAPH_LIMIT}" \
  --attempts "${GRAPH_ATTEMPTS}" \
  --tensor-parallel-size 1 \
  --seed 42

sha256sum "${run_dir}/graphs.jsonl" "${run_dir}/manifest.json" \
  "${run_dir}/source-commit.txt" > "${run_dir}/artifact-manifest.sha256"
printf 'GRAF_GRAPH_CACHE_MANIFEST=%s\n' "${run_dir}/manifest.json"
