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
configured_save_steps="$(python3 - "${CH_CANDIDATE_CONFIG}" <<'PY'
from opsd_research.config import load_config
import sys
print(load_config(sys.argv[1]).data["save_steps"])
PY
)"
configured_gpus="$(python3 - "${CH_CANDIDATE_CONFIG}" <<'PY'
from opsd_research.config import load_config
import sys
print(load_config(sys.argv[1]).data["num_gpus"])
PY
)"
configured_accumulation="$(python3 - "${CH_CANDIDATE_CONFIG}" <<'PY'
from opsd_research.config import load_config
import sys
print(load_config(sys.argv[1]).data["gradient_accumulation_steps"])
PY
)"
max_sequence_length="$(python3 - "${CH_CANDIDATE_CONFIG}" <<'PY'
from opsd_research.config import load_config
import sys
print(load_config(sys.argv[1]).data.get("max_sequence_length", 28672))
PY
)"
configured_vllm_utilization="$(python3 - "${CH_CANDIDATE_CONFIG}" <<'PY'
from opsd_research.config import load_config
import sys
print(load_config(sys.argv[1]).data.get("vllm_gpu_memory_utilization", 0.35))
PY
)"
if [[ "${CH_MAX_STEPS}" == 1 || "${CH_MAX_STEPS}" == 5 ]]; then
  export GRAF_SMOKE_MAX_STEPS="${CH_MAX_STEPS}"
  effective_save_steps="${CH_MAX_STEPS}"
elif [[ "${CH_MAX_STEPS}" != "${configured_steps}" ]]; then
  echo "CH_MAX_STEPS must be 1, 5, or the configured ${configured_steps}" >&2
  exit 2
else
  effective_save_steps="${configured_save_steps}"
fi
if [[ -n "${SLURM_GPUS_ON_NODE:-}" &&
      "${SLURM_GPUS_ON_NODE}" != "${configured_gpus}" ]]; then
  echo "allocated ${SLURM_GPUS_ON_NODE} GPUs but config requires ${configured_gpus}" >&2
  exit 2
fi

export GRAF_CONFIG="${PROJECT_SOURCE}/${CH_CANDIDATE_CONFIG}"
export CH_DOSSIER_MANIFEST
export OPSD_DATASET_REVISION=1435fb21d4fecc8ad4966a26f22a874cf2b527f1
export OPSD_EXACT_JSD_VOCAB_CHUNK_SIZE=4096
export OPSD_TAIL_LOGITS_ONLY=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
vllm_utilization="${CH_VLLM_GPU_MEMORY_UTILIZATION:-${configured_vllm_utilization}}"
run_name="${CH_RUN_NAME:-ch-${CH_CANDIDATE_ID}-${CH_MAX_STEPS}-${SLURM_JOB_ID}}"
output_dir="${TRAIN_ROOT}/${run_name}"
mkdir -p "${output_dir}"
cp "${CH_CANDIDATE_CONFIG}" "${output_dir}/config.yaml"
cp "${CH_DOSSIER_MANIFEST}" "${output_dir}/ch-dossier-manifest.json"
export OPSD_TRAINING_PARTITION_MANIFEST="${output_dir}/training-partition.json"
printf 'candidate=%s\nmax_steps=%s\nnum_gpus=%s\ngradient_accumulation_steps=%s\neffective_batch_size=32\nvllm_gpu_memory_utilization=%s\npytorch_cuda_alloc_conf=%s\n' \
  "${CH_CANDIDATE_ID}" "${CH_MAX_STEPS}" "${configured_gpus}" \
  "${configured_accumulation}" "${vllm_utilization}" \
  "${PYTORCH_CUDA_ALLOC_CONF}" \
  > "${output_dir}/runtime-overrides.txt"
record_run_manifest \
  "${output_dir}" \
  runtime-overrides.txt \
  ch-dossier-manifest.json

start_gpu_telemetry "${run_name}"
trap stop_gpu_telemetry EXIT
canonical_training_log="${output_dir}/training.log"
accelerate launch \
  --config_file third_party/opsd/accelerate.yaml \
  --num_processes "${configured_gpus}" \
  --gradient_accumulation_steps "${configured_accumulation}" \
  --main_process_port "${CH_PORT}" \
  -m opsd_research.launch_graf_opsd \
  --model_name_or_path Qwen/Qwen3-4B \
  --model_revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --student_model_revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --learning_rate 5e-6 \
  --max_grad_norm 0.1 \
  --per_device_train_batch_size 1 \
  --gradient_checkpointing \
  --gradient_accumulation_steps "${configured_accumulation}" \
  --output_dir "${TRAIN_ROOT}" \
  --run_config "${run_name}" \
  --num_train_epochs 30 \
  --max_steps "${CH_MAX_STEPS}" \
  --max_completion_length "${completion_length}" \
  --save_steps "${effective_save_steps}" \
  --logging_steps 1 \
  --attn_implementation flash_attention_2 \
  --torch_dtype bfloat16 \
  --max_length "${max_sequence_length}" \
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
  --wandb_project CH-OPSD-autoresearch \
  2>&1 | tee -a "${canonical_training_log}"

stop_gpu_telemetry
trap - EXIT
combined_telemetry="${output_dir}/gpu-telemetry.csv"
telemetry_files=("${TELEMETRY_ROOT}/${run_name}-"*.csv)
if (( ${#telemetry_files[@]} == 0 )); then
  echo "no telemetry files found for ${run_name}" >&2
  exit 1
fi
awk 'FNR == 1 && NR != 1 { } { print }' "${telemetry_files[@]}" \
  > "${combined_telemetry}"

python3 -m opsd_research.summarize_training \
  --training-dir "${output_dir}" \
  --training-log "${canonical_training_log}" \
  --telemetry "${combined_telemetry}" \
  --output "${output_dir}/training-summary.json" \
  --max-completion-length "${completion_length}"
