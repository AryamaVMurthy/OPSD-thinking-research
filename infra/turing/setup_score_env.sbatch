#!/usr/bin/env bash
#SBATCH --job-name=opsd-score-setup
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node10
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module load u22/cuda/12.4
source "${HOME}/OPSD-thinking-research/infra/turing/common.sh"

UV_BIN="${HOME}/.local/bin/uv"
export UV_CACHE_DIR="${SCRATCH_ROOT}/uv-cache"
export UV_PYTHON_INSTALL_DIR="${SCRATCH_ROOT}/uv-python"
"${UV_BIN}" python install 3.12
if [[ ! -x "${SCORE_ENV_DIR}/bin/python" ]]; then
  "${UV_BIN}" venv --python 3.12 "${SCORE_ENV_DIR}"
fi
"${UV_BIN}" pip install --python "${SCORE_ENV_DIR}/bin/python" \
  antlr4-python3-runtime==4.11.0 loguru==0.7.3 PyYAML==6.0.2 \
  regex==2024.11.6 sympy==1.13.1
"${UV_BIN}" pip install --python "${SCORE_ENV_DIR}/bin/python" --no-deps -e "${PROJECT_SOURCE}"
"${SCORE_ENV_DIR}/bin/python" - <<'PY'
from opsd_research.matharena_grading import grade_response

result = grade_response(r"\boxed{26!}", r"2^{25} \cdot 26!", strict_parsing=False)
assert result["correct"] is False
print("score-environment-ready")
PY
