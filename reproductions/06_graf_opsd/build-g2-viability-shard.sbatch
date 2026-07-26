#!/usr/bin/env bash
# An independently auditable viability target build.  The eight one-GPU
# shards partition every accepted graph in the immutable cache; no fixed
# number of graphs per shard is assumed.
# Submit with: sbatch --array=0-7 --export=ALL,CACHE_BUILD_JOB_ID=<id> ...
#SBATCH --job-name=graf-g2-viability
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node03
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail
module load u22/cuda/12.4
source "${HOME}/OPSD-thinking-research/infra/turing/common.sh"
cd "${PROJECT_SOURCE}"

: "${CACHE_BUILD_JOB_ID:?missing immutable graph-cache build job ID}"
: "${SLURM_ARRAY_TASK_ID:?this job must be submitted as an eight-task array}"
: "${G2_SHARD_COUNT:=8}"
: "${VIABILITY_CACHE_LABEL:=g2-viability}"
: "${GRAF_RECOVERY_CONDITIONED:=0}"
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= G2_SHARD_COUNT )); then
  echo "G2 viability task index is outside the declared shard count" >&2
  exit 2
fi

graph_manifest="${SCRATCH_ROOT}/graf_cache/g1-${CACHE_BUILD_JOB_ID}/manifest.json"
target_root="${SCRATCH_ROOT}/graf_cache/${VIABILITY_CACHE_LABEL}-${SLURM_ARRAY_JOB_ID}"
mkdir -p "${target_root}"
accepted_graphs="$(python3 - "${graph_manifest}" <<'PY'
import json, sys
manifest=json.load(open(sys.argv[1], encoding="utf-8"))
print(int(manifest["accepted_examples"]))
PY
)"
if (( accepted_graphs < G2_SHARD_COUNT )); then
  echo "graph cache has ${accepted_graphs} accepted records, fewer than ${G2_SHARD_COUNT} shards" >&2
  exit 2
fi
records_per_shard=$(( (accepted_graphs + G2_SHARD_COUNT - 1) / G2_SHARD_COUNT ))
start_index=$(( SLURM_ARRAY_TASK_ID * records_per_shard ))
remaining=$(( accepted_graphs - start_index ))
limit=$(( remaining < records_per_shard ? remaining : records_per_shard ))
output="${target_root}/part-${SLURM_ARRAY_TASK_ID}.jsonl"
recovery_args=()
if [[ "${GRAF_RECOVERY_CONDITIONED}" == 1 ]]; then
  recovery_args+=(--recovery-conditioned)
elif [[ "${GRAF_RECOVERY_CONDITIONED}" != 0 ]]; then
  echo "GRAF_RECOVERY_CONDITIONED must be 0 or 1" >&2
  exit 2
fi

python3 -m opsd_research.build_graf_viability \
  --graph-manifest "${graph_manifest}" \
  --output "${output}" \
  --start-index "${start_index}" --limit "${limit}" \
  --samples-per-action 2 --temperature 1.0 \
  --model Qwen/Qwen3-4B \
  --model-revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --tensor-parallel-size 1 --seed 42 "${recovery_args[@]}"
