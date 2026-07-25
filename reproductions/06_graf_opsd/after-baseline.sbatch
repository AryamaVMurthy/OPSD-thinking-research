#!/usr/bin/env bash
#SBATCH --job-name=graf-after-base
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node10
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module load u22/cuda/12.4
source "${HOME}/OPSD-thinking-research/infra/turing/common.sh"
cd "${PROJECT_SOURCE}"

git fetch origin
git merge --ff-only origin/codex/thinking-opsd-reproduction
cache_job="$(GRAPH_LIMIT="${GRAPH_LIMIT:-32}" sbatch --parsable reproductions/06_graf_opsd/build-graph-cache.sbatch)"
echo "graph_cache_job=${cache_job}"
sbatch --parsable --dependency="afterok:${cache_job}" \
  --export="ALL,CACHE_BUILD_JOB_ID=${cache_job}" \
  reproductions/06_graf_opsd/after-cache.sbatch
