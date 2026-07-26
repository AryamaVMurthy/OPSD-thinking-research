#!/usr/bin/env bash
# Fresh 200-step confirmation after a candidate clears the 50-step AIME24 gate.
#SBATCH --job-name=graf-promoted-full
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
: "${CANDIDATE_ID:?}" "${CANDIDATE_CONFIG:?}" "${METHOD:?}" "${COMPLETION_LENGTH:?}"
python3 -m opsd_research.cli validate-config "${CANDIDATE_CONFIG}"
export GRAF_CONFIG="${PROJECT_SOURCE}/${CANDIDATE_CONFIG}"
export GRAF_FULL_MAX_STEPS=200 OPSD_DATASET_REVISION=1435fb21d4fecc8ad4966a26f22a874cf2b527f1 OPSD_EXACT_JSD_VOCAB_CHUNK_SIZE=4096 OPSD_TAIL_LOGITS_ONLY=1
run_name="graf-${CANDIDATE_ID}-200-${SLURM_JOB_ID}"
output_dir="${TRAIN_ROOT}/${run_name}"
mkdir -p "${output_dir}"
export OPSD_TRAINING_PARTITION_MANIFEST="${output_dir}/training-partition.json"
cp "${CANDIDATE_CONFIG}" "${output_dir}/config.yaml"
printf 'candidate=%s\nmax_steps=200\nconfirmation=fresh_from_base\n' "${CANDIDATE_ID}" > "${output_dir}/runtime-overrides.txt"
files=(runtime-overrides.txt)
if [[ -n "${GRAF_GRAPH_CACHE_MANIFEST:-}" ]]; then cp "${GRAF_GRAPH_CACHE_MANIFEST}" "${output_dir}/graph-cache-manifest.json"; files+=(graph-cache-manifest.json); fi
if [[ -n "${GRAF_VIABILITY_MANIFEST:-}" ]]; then cp "${GRAF_VIABILITY_MANIFEST}" "${output_dir}/viability-manifest.json"; files+=(viability-manifest.json); fi
record_run_manifest "${output_dir}" "${files[@]}"
start_gpu_telemetry "${run_name}"; trap stop_gpu_telemetry EXIT
accelerate launch --config_file third_party/opsd/accelerate.yaml --num_processes 4 --gradient_accumulation_steps 8 --main_process_port 12974 \
  -m opsd_research.launch_graf_opsd --model_name_or_path Qwen/Qwen3-4B --model_revision 1cfa9a7208912126459214e8b04321603b3df60c --student_model_revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --learning_rate 5e-6 --max_grad_norm 0.1 --per_device_train_batch_size 1 --gradient_checkpointing --gradient_accumulation_steps 8 --output_dir "${TRAIN_ROOT}" --run_config "${run_name}" --num_train_epochs 30 --max_steps 200 --max_completion_length "${COMPLETION_LENGTH}" --save_steps 200 --logging_steps 1 \
  --attn_implementation flash_attention_2 --torch_dtype bfloat16 --max_length 20000 --beta 0 --use_vllm --vllm_mode colocate --vllm_gpu_memory_utilization "${VLLM_GPU_UTILIZATION:-0.6}" --vllm_tensor_parallel_size 1 --use_peft --lora_r 64 --lora_alpha 128 --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj --temperature 1.1 --top_p 0.95 --top_k 20 --lmbda 1 --fixed_teacher --student_thinking --teacher_thinking --jsd_token_clip 0.05 --seed 42 --data_seed 42 --report_to wandb --wandb_project GRAF-OPSD-autoresearch
python3 -m opsd_research.summarize_training --training-dir "${output_dir}" --training-log "${PROJECT_SOURCE}/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" --telemetry "${TELEMETRY_ROOT}/${run_name}-${SLURM_JOB_ID}.csv" --output "${output_dir}/training-summary.json" --max-completion-length "${COMPLETION_LENGTH}"
