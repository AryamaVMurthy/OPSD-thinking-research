#!/usr/bin/env bash
#SBATCH --job-name=opsd-rescore
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node10
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module load u22/cuda/12.4
source "${HOME}/OPSD-thinking-research/infra/turing/common.sh"

: "${CONFIG:?set CONFIG to a repository-relative math_eval YAML}"
: "${RUN_NAME:?set RUN_NAME to the existing result directory name}"
output_dir="${RESULTS_ROOT}/${RUN_NAME}"
test -d "${output_dir}"
test -f "${PROJECT_SOURCE}/${CONFIG}"
python3 -m opsd_research.cli validate-config "${PROJECT_SOURCE}/${CONFIG}"

shopt -s nullglob
raw=("${output_dir}"/generations.shard*.jsonl)
if (( ${#raw[@]} != 8 )); then
  echo "Expected eight complete generation shards, found ${#raw[@]}" >&2
  exit 2
fi
score_python="${SCORE_ENV_DIR}/bin/python"
if [[ ! -x "${score_python}" ]]; then
  score_python="${ENV_DIR}/bin/python"
  echo "{\"event\":\"score_env_fallback\",\"python\":\"${score_python}\"}" >&2
fi
"${score_python}" -m opsd_research.rescore_matharena \
  --config "${PROJECT_SOURCE}/${CONFIG}" \
  --input "${raw[@]}" \
  --grades-output "${output_dir}/official-grades.jsonl" \
  --summary-output "${output_dir}/summary.json"
"${PROJECT_SOURCE}/infra/turing/artifact_manifest.sh" \
  "${output_dir}" "artifact-manifest-rescore-${SLURM_JOB_ID}.sha256"
