#!/usr/bin/env bash
#SBATCH --job-name=opsd-smoke-infer
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node10
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module load u22/cuda/12.4
source "${HOME}/OPSD-thinking-research/infra/turing/common.sh"
cd "${PROJECT_SOURCE}"

smoke_root="${RESULTS_ROOT}/smoke-inference"
mkdir -p "${smoke_root}"

for model_size in 1p7b 4b; do
  config="reproductions/01_qwen3_thinking_math/configs/qwen3-${model_size}-aime25.yaml"
  output="${smoke_root}/${model_size}.jsonl"
  python3 -m opsd_research.math_eval \
    --config "${config}" \
    --output "${output}" \
    --method smoke \
    --checkpoint none \
    --limit 1
  python3 -m opsd_research.verify_smoke --input "${output}"
done
