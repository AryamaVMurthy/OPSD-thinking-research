#!/usr/bin/env bash
# Answer-free Fisher-space specificity audit for the completed FiNOD run.
#SBATCH --job-name=finod-specificity
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node03
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home/aryama.murthy/finod-control/finod-specificity-%j.out

set -euo pipefail

module load u22/cuda/12.4
export PATH="${HOME}/.local/bin:${PATH}"
export UV_CONCURRENT_DOWNLOADS=1
export UV_CONCURRENT_BUILDS=1
export UV_CONCURRENT_INSTALLS=1
export UV_LINK_MODE=copy

task_root="/scratch/${USER}/opsd-thinking-research/finod-specificity-${SLURM_JOB_ID}"
source_dir="${task_root}/source"
input_dir="${task_root}/input"
mkdir -p "${source_dir}" "${input_dir}"
tar -xzf "${HOME}/finod-control/finod-specificity-source.tgz" -C "${source_dir}"
tar -xzf "${HOME}/finod-control/finod-specificity-input.tgz" -C "${input_dir}"

export HF_HOME="/scratch/${USER}/opsd-thinking-research/hf"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export UV_CACHE_DIR="/scratch/${USER}/opsd-thinking-research/uv-cache"
export PYTHONUNBUFFERED=1

cd "${source_dir}"
nvidia-smi
uv run python -m opsd_research.analyze_finod_guide_specificity \
  --graphs-jsonl "${input_dir}/finod/cache/representative-17172/graphs.jsonl" \
  --generations-dir \
    "${input_dir}/training/finod-problem-only-1024-full32-seeded/generations" \
  --samples-per-group 6 \
  --positions 4 \
  --output "${task_root}/finod-guide-specificity.json"

cp "${task_root}/finod-guide-specificity.json" \
  "${HOME}/finod-control/finod-guide-specificity-${SLURM_JOB_ID}.json"
