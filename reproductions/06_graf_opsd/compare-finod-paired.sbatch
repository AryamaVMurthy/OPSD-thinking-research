#!/usr/bin/env bash
# Produce a paired, official-grade comparison after matched FiNOD evaluations.
#SBATCH --job-name=compare-finod
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node10
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail

: "${PROJECT_SOURCE:?}"
: "${BASELINE_RUN:?}"
: "${TREATMENT_RUN:?}"
: "${SAMPLES_PER_PROBLEM:?}"

source "${HOME}/OPSD-thinking-research/infra/turing/common.sh"
cd "${PROJECT_SOURCE}"
baseline="${RESULTS_ROOT}/${BASELINE_RUN}"
treatment="${RESULTS_ROOT}/${TREATMENT_RUN}"

python3 -m opsd_research.compare_math \
  --baseline-input "${baseline}"/generations.shard*.jsonl \
  --baseline-grades "${baseline}/official-grades.jsonl" \
  --treatment-input "${treatment}"/generations.shard*.jsonl \
  --treatment-grades "${treatment}/official-grades.jsonl" \
  --samples-per-problem "${SAMPLES_PER_PROBLEM}" \
  --output "${treatment}/paired-comparison.json"
