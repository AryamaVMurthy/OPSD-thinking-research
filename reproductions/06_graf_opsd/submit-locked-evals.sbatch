#!/usr/bin/env bash
# Submit the untouched locked AIME25/AIME26 evaluations after a development
# candidate has cleared its pre-registered paired AIME24 promotion gate.
#SBATCH --job-name=graf-submit-locked
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node10
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module load u22/cuda/12.4
source "${HOME}/OPSD-thinking-research/infra/turing/common.sh"
cd "${PROJECT_SOURCE}"

: "${METHOD:?missing promoted method name}"
: "${ADAPTER:?missing promoted adapter path}"
test -f "${ADAPTER}/adapter_model.safetensors"
for benchmark in aime25 aime26; do
  config="reproductions/01_qwen3_thinking_math/configs/qwen3-4b-${benchmark}.yaml"
  run_name="${METHOD}-${benchmark}-${SLURM_JOB_ID}"
  job="$(CONFIG="${config}" EVAL_MODULE=math_eval RUN_NAME="${run_name}" \
    METHOD="${METHOD}" CHECKPOINT=promoted ADAPTER="${ADAPTER}" \
    sbatch --parsable infra/turing/run_eval.sbatch)"
  echo "${benchmark}_eval_job=${job} run=${run_name}"
done
