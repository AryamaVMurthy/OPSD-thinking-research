#!/usr/bin/env bash
#SBATCH --job-name=opsd-setup
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node10
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module load u22/cuda/12.4
source "${HOME}/OPSD-thinking-research/infra/turing/common.sh"

UV_BIN="${HOME}/.local/bin/uv"
export UV_CACHE_DIR="${SCRATCH_ROOT}/uv-cache"
export UV_PYTHON_INSTALL_DIR="${SCRATCH_ROOT}/uv-python"

"${UV_BIN}" python install 3.10
if [[ ! -x "${ENV_DIR}/bin/python3" ]]; then
  "${UV_BIN}" venv --python 3.10 "${ENV_DIR}"
fi
source "${ENV_DIR}/bin/activate"

"${UV_BIN}" pip install \
  torch==2.8.0 \
  accelerate==1.11.0 \
  transformers==4.57.1 \
  trl==0.26.0 \
  datasets==3.6.0 \
  deepspeed==0.18.2 \
  peft==0.17.1 \
  bitsandbytes==0.48.2 \
  wandb==0.22.3 \
  vllm==0.11.0 \
  xformers==0.0.32.post1 \
  triton==3.4.0 \
  einops==0.8.1 \
  safetensors==0.5.3 \
  sentencepiece==0.1.99 \
  tiktoken==0.9.0 \
  math-verify==0.8.0 \
  PyYAML==6.0.2 \
  pebble==5.1.3 \
  ninja==1.13.0 \
  packaging==25.0

"${UV_BIN}" pip install --no-build-isolation flash-attn==2.8.3
"${UV_BIN}" pip install --no-deps -e "${PROJECT_SOURCE}"

python3 -m opsd_research.cli validate-all
python3 -m opsd_research.prepare_assets
python3 -m opsd_research.smoke
"${UV_BIN}" pip freeze > "${SCRATCH_ROOT}/environment-freeze.txt"
df -h "${SCRATCH_ROOT}"
