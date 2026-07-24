#!/usr/bin/env bash
#SBATCH --job-name=opsd-gpu-bind
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node10
#SBATCH --gres=gpu:8
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module load u22/cuda/12.4

srun \
  --kill-on-bad-exit=1 \
  --gpus-per-task=1 \
  --gpu-bind=single:1 \
  bash -c '
    set -euo pipefail
    gpu_uuid="$(nvidia-smi --query-gpu=uuid --format=csv,noheader)"
    printf "rank=%s visible=%s uuid=%s\n" \
      "${SLURM_PROCID}" "${CUDA_VISIBLE_DEVICES}" "${gpu_uuid}"
  '
