#!/usr/bin/env bash
# Audited four-GPU runner for CH-OPSD smoke and bounded development training.
#SBATCH --job-name=ch-opsd-train
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node03
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=350G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module load u22/cuda/12.4
source "${HOME}/OPSD-thinking-research/infra/turing/common.sh"
cd "${PROJECT_SOURCE}"

: "${CH_DOSSIER_MANIFEST:?}" "${CH_CANDIDATE_CONFIG:?}"
: "${CH_CANDIDATE_ID:?}" "${CH_MAX_STEPS:?}" "${CH_PORT:?}"
python3 -m opsd_research.cli validate-config "${CH_CANDIDATE_CONFIG}"
completion_length="$(python3 - "${CH_CANDIDATE_CONFIG}" <<'PY'
from opsd_research.config import load_config
import sys
print(load_config(sys.argv[1]).data["max_completion_length"])
PY
)"
configured_steps="$(python3 - "${CH_CANDIDATE_CONFIG}" <<'PY'
from opsd_research.config import load_config
import sys
print(load_config(sys.argv[1]).data["max_steps"])
PY
)"
if [[ "${CH_MAX_STEPS}" == 5 ]]; then
  export GRAF_SMOKE_MAX_STEPS=5
elif [[ "${CH_MAX_STEPS}" != "${configured_steps}" ]]; then
  echo "CH_MAX_STEPS must be 5 or the configured ${configured_steps}" >&2
  exit 2
fi

export GRAF_CONFIG="${PROJECT_SOURCE}/${CH_CANDIDATE_CONFIG}"
export CH_DOSSIER_MANIFEST
export OPSD_DATASET_REVISION=1435fb21d4fecc8ad4966a26f22a874cf2b527f1
export OPSD_EXACT_JSD_VOCAB_CHUNK_SIZE=4096
export OPSD_TAIL_LOGITS_ONLY=1
vllm_utilization="${CH_VLLM_GPU_MEMORY_UTILIZATION:-0.35}"
run_name="ch-${CH_CANDIDATE_ID}-${CH_MAX_STEPS}-${SLURM_JOB_ID}"
output_dir="${TRAIN_ROOT}/${run_name}"
mkdir -p "${output_dir}"
cp "${CH_CANDIDATE_CONFIG}" "${output_dir}/config.yaml"
cp "${CH_DOSSIER_MANIFEST}" "${output_dir}/ch-dossier-manifest.json"
export OPSD_TRAINING_PARTITION_MANIFEST="${output_dir}/training-partition.json"
printf 'candidate=%s\nmax_steps=%s\nnum_gpus=4\ngradient_accumulation_steps=8\neffective_batch_size=32\nvllm_gpu_memory_utilization=%s\n' \
  "${CH_CANDIDATE_ID}" "${CH_MAX_STEPS}" "${vllm_utilization}" \
  > "${output_dir}/runtime-overrides.txt"
record_run_manifest \
  "${output_dir}" \
  runtime-overrides.txt \
  ch-dossier-manifest.json

start_gpu_telemetry "${run_name}"
trap stop_gpu_telemetry EXIT
accelerate launch \
  --config_file third_party/opsd/accelerate.yaml \
  --num_processes 4 \
  --gradient_accumulation_steps 8 \
  --main_process_port "${CH_PORT}" \
  -m opsd_research.launch_graf_opsd \
  --model_name_or_path Qwen/Qwen3-4B \
  --model_revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --student_model_revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --learning_rate 5e-6 \
  --max_grad_norm 0.1 \
  --per_device_train_batch_size 1 \
  --gradient_checkpointing \
  --gradient_accumulation_steps 8 \
  --output_dir "${TRAIN_ROOT}" \
  --run_config "${run_name}" \
  --num_train_epochs 30 \
  --max_steps "${CH_MAX_STEPS}" \
  --max_completion_length "${completion_length}" \
  --save_steps "${CH_MAX_STEPS}" \
  --logging_steps 1 \
  --attn_implementation flash_attention_2 \
  --torch_dtype bfloat16 \
  --max_length 20000 \
  --beta 0 \
  --use_vllm \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization "${vllm_utilization}" \
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
  --wandb_project CH-OPSD-autoresearch

python3 -m opsd_research.summarize_training \
  --training-dir "${output_dir}" \
  --training-log "${PROJECT_SOURCE}/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" \
  --telemetry "${TELEMETRY_ROOT}/${run_name}-${SLURM_JOB_ID}.csv" \
  --output "${output_dir}/training-summary.json" \
  --max-completion-length "${completion_length}"
