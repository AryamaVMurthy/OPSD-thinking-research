#!/usr/bin/env bash
# A lightweight readiness gate for the AIME-only autoresearch loop.
# The broad setup may be used for the full benchmark suite; this gate avoids
# waiting for code-evaluation assets before math-only C1/GRAF experiments.
#SBATCH --job-name=opsd-aime-ready
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node03
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module load u22/cuda/12.4
source "${HOME}/OPSD-thinking-research/infra/turing/common.sh"
cd "${PROJECT_SOURCE}"

test -x "${ENV_DIR}/bin/python3"
test -x "${SCORE_ENV_DIR}/bin/python3"
python3 -m opsd_research.cli validate-config reproductions/06_graf_opsd/configs/c1-long-rollout-4096.yaml
python3 -m opsd_research.prepare_assets --scope aime
python3 -m opsd_research.smoke
"${SCORE_ENV_DIR}/bin/python" -c \
  "from opsd_research.matharena_grading import grade_response; assert grade_response(r'\\boxed{26!}', r'2^{25} \\cdot 26!', strict_parsing=False)['correct'] is False"
UV_BIN="${HOME}/.local/bin/uv"
"${UV_BIN}" pip freeze > "${SCRATCH_ROOT}/environment-freeze.txt"
"${UV_BIN}" pip freeze --python "${SCORE_ENV_DIR}/bin/python" > "${SCRATCH_ROOT}/score-environment-freeze.txt"
