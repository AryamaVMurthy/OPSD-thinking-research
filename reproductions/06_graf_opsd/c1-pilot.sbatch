#!/usr/bin/env bash
# Development pilot for the 4,096-token no-graph OPSD control.
#SBATCH --job-name=graf-c1-pilot
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node03
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=350G
#SBATCH --time=18:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module load u22/cuda/12.4
source "${HOME}/OPSD-thinking-research/infra/turing/common.sh"
cd "${PROJECT_SOURCE}"
CONFIG="reproductions/06_graf_opsd/configs/c1-long-rollout-4096.yaml"
python3 -m opsd_research.cli validate-config "${CONFIG}"
export GRAF_CONFIG="${PROJECT_SOURCE}/${CONFIG}"
export OPSD_DATASET_REVISION=1435fb21d4fecc8ad4966a26f22a874cf2b527f1
export OPSD_EXACT_JSD_VOCAB_CHUNK_SIZE=4096
export OPSD_TAIL_LOGITS_ONLY=1
run_name="graf-c1-long-4096-50-${SLURM_JOB_ID}"
output_dir="${TRAIN_ROOT}/${run_name}"
mkdir -p "${output_dir}"
cp "${CONFIG}" "${output_dir}/config.yaml"
printf 'candidate=c1-long-rollout-4096\nmax_steps=50\nnum_gpus=4\ngradient_accumulation_steps=8\neffective_batch_size=32\n' > "${output_dir}/runtime-overrides.txt"
record_run_manifest "${output_dir}" runtime-overrides.txt
start_gpu_telemetry "${run_name}"
trap stop_gpu_telemetry EXIT
accelerate launch \
  --config_file third_party/opsd/accelerate.yaml --num_processes 4 \
  --gradient_accumulation_steps 8 --main_process_port 12973 \
  -m opsd_research.launch_graf_opsd \
  --model_name_or_path Qwen/Qwen3-4B \
  --model_revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --student_model_revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --learning_rate 5e-6 --max_grad_norm 0.1 \
  --per_device_train_batch_size 1 --gradient_checkpointing --gradient_accumulation_steps 8 \
  --output_dir "${TRAIN_ROOT}" --run_config "${run_name}" --num_train_epochs 30 \
  --max_steps 50 --max_completion_length 4096 --save_steps 50 --logging_steps 1 \
  --attn_implementation flash_attention_2 --torch_dtype bfloat16 --max_length 20000 \
  --beta 0 --use_vllm --vllm_mode colocate --vllm_gpu_memory_utilization 0.55 \
  --vllm_tensor_parallel_size 1 --use_peft --lora_r 64 --lora_alpha 128 \
  --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --temperature 1.1 --top_p 0.95 --top_k 20 --lmbda 1 --fixed_teacher \
  --student_thinking --teacher_thinking --jsd_token_clip 0.05 --seed 42 --data_seed 42 \
  --report_to wandb --wandb_project GRAF-OPSD-autoresearch
python3 -m opsd_research.summarize_training \
  --training-dir "${output_dir}" \
  --training-log "${PROJECT_SOURCE}/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" \
  --telemetry "${TELEMETRY_ROOT}/${run_name}-${SLURM_JOB_ID}.csv" \
  --output "${output_dir}/training-summary.json" --max-completion-length 4096
