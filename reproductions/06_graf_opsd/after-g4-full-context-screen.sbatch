#!/usr/bin/env bash
# Reuse G4's first four official samples against a four-sample long baseline.
#SBATCH --job-name=after-g4-long4
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node03
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module load u22/cuda/12.4
source "${HOME}/OPSD-thinking-research/infra/turing/common.sh"
cd "${PROJECT_SOURCE}"
baseline="${RESULTS_ROOT}/graf-aime24-base-full-context-screen-20260727"
treatment="${RESULTS_ROOT}/graf-g4-viability-long-4096-50-aime24-16356"
test -f "${baseline}/summary.json"
test -f "${treatment}/summary.json"
python3 -m opsd_research.compare_math \
  --baseline-input "${baseline}"/generations.shard*.jsonl \
  --baseline-grades "${baseline}/official-grades.jsonl" \
  --treatment-input "${treatment}"/generations.shard*.jsonl \
  --treatment-grades "${treatment}/official-grades.jsonl" \
  --samples-per-problem 4 \
  --take-first-samples \
  --output "${treatment}/paired-full-context-vs-base.json"
python3 -m opsd_research.graf_development_report \
  --comparison "${treatment}/paired-full-context-vs-base.json" \
  --candidate-id g4-viability-long-4096 \
  --output "${treatment}/full-context-development-report.md"
