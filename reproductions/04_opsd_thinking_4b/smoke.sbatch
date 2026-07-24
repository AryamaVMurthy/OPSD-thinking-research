#!/usr/bin/env bash
#SBATCH --job-name=opsd-4b-smoke
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node10
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=768G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module load u22/cuda/12.4
source "${HOME}/OPSD-thinking-research/infra/turing/common.sh"
cd "${PROJECT_SOURCE}"

CONFIG="reproductions/04_opsd_thinking_4b/configs/train.yaml"
python3 -m opsd_research.cli validate-config "${CONFIG}"
export OPSD_DATASET_REVISION=1435fb21d4fecc8ad4966a26f22a874cf2b527f1
export OPSD_EXACT_JSD_VOCAB_CHUNK_SIZE=4096
export OPSD_SMOKE_MAX_STEPS=5

run_name="qwen3-4b-opsd-thinking-smoke-${SLURM_JOB_ID}"
output_dir="${TRAIN_ROOT}/${run_name}"
mkdir -p "${output_dir}"
cp "${CONFIG}" "${output_dir}/protocol.yaml"
cp "${CONFIG}" "${output_dir}/config.yaml"
printf 'run_type=memory_smoke\nmax_steps=5\nsave_steps=5\n' \
  > "${output_dir}/runtime-overrides.txt"
record_run_manifest "${output_dir}"
(
  cd "${output_dir}"
  sha256sum runtime-overrides.txt >> "manifest-${SLURM_JOB_ID}.sha256"
)

start_gpu_telemetry "${run_name}"
trap stop_gpu_telemetry EXIT

accelerate launch \
  --config_file third_party/opsd/accelerate.yaml \
  --num_processes 8 \
  --gradient_accumulation_steps 4 \
  --main_process_port 12949 \
  -m opsd_research.launch_official_opsd \
  --model_name_or_path Qwen/Qwen3-4B \
  --model_revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --student_model_revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --learning_rate 5e-6 \
  --max_grad_norm 0.1 \
  --per_device_train_batch_size 1 \
  --gradient_checkpointing \
  --gradient_accumulation_steps 4 \
  --output_dir "${TRAIN_ROOT}" \
  --run_config "${run_name}" \
  --num_train_epochs 30 \
  --max_steps 5 \
  --max_completion_length 1024 \
  --save_steps 5 \
  --logging_steps 1 \
  --attn_implementation flash_attention_2 \
  --torch_dtype bfloat16 \
  --max_length 20000 \
  --beta 0 \
  --use_vllm \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization 0.6 \
  --vllm_tensor_parallel_size 1 \
  --use_peft \
  --lora_r 64 \
  --lora_alpha 128 \
  --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --temperature 1.1 \
  --top_p 0.95 \
  --top_k 20 \
  --lmbda 1 \
  --fixed_teacher \
  --student_thinking \
  --teacher_thinking \
  --jsd_token_clip 0.05 \
  --seed 42 \
  --data_seed 42 \
  --report_to wandb \
  --wandb_project OPSD-thinking-reproduction

python3 -m opsd_research.summarize_training \
  --training-dir "${output_dir}" \
  --training-log "${PROJECT_SOURCE}/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" \
  --telemetry "${TELEMETRY_ROOT}/${run_name}-${SLURM_JOB_ID}.csv" \
  --output "${output_dir}/training-summary.json" \
  --max-completion-length 1024
