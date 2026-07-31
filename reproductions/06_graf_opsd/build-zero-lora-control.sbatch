#!/usr/bin/env bash
# Build a zero-scale adapter for a same-runtime LoRA placebo evaluation.
#SBATCH --job-name=zero-lora-control
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node10
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail

: "${PROJECT_SOURCE:?}"
: "${SOURCE_ADAPTER:?}"
: "${OUTPUT_ADAPTER:?}"

source "${HOME}/OPSD-thinking-research/infra/turing/common.sh"
cd "${PROJECT_SOURCE}"
test -z "$(git status --short)"
python3 -m opsd_research.zero_lora_control \
  --source "${SOURCE_ADAPTER}" \
  --output "${OUTPUT_ADAPTER}"
