#!/usr/bin/env bash
#SBATCH --job-name=opsd-lcb-score
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node10
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
source "${HOME}/OPSD-thinking-research/infra/turing/common.sh"
: "${CONFIG:?set CONFIG}"
: "${RUN_NAME:?set RUN_NAME}"

OUTPUT_DIR="${RESULTS_ROOT}/${RUN_NAME}"
python3 -m opsd_research.score_lcb \
  --config "${PROJECT_SOURCE}/${CONFIG}" \
  --input "${OUTPUT_DIR}"/generations.shard*.jsonl \
  --graded-output "${OUTPUT_DIR}/graded.jsonl" \
  --output "${OUTPUT_DIR}/summary.json" \
  --processes 32
