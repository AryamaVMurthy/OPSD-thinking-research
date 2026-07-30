#!/usr/bin/env bash
# Audited eight-A100 runner for a FiNOD smoke or development pilot.
#SBATCH --job-name=finod
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node10
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=900G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail

module load u22/cuda/12.4
source "${HOME}/OPSD-thinking-research/infra/turing/common.sh"
cd "${PROJECT_SOURCE}"

: "${FINOD_CONFIG:?}"
: "${FINOD_MAX_STEPS:?}"
: "${FINOD_PORT:?}"
: "${GRAF_GRAPH_CACHE_MANIFEST:?}"

python3 -m opsd_research.cli validate-config "${FINOD_CONFIG}"
python3 - "${FINOD_CONFIG}" <<'PY'
from opsd_research.config import load_config
import sys

config = load_config(sys.argv[1]).data
if config["graph_mode"] != "finod_scaffold":
    raise SystemExit("FiNOD runner requires graph_mode=finod_scaffold")
if config["num_gpus"] != 8 or config["gradient_accumulation_steps"] != 4:
    raise SystemExit("FiNOD runner requires the registered DP8/GA4 recipe")
PY

completion_length="$(python3 - "${FINOD_CONFIG}" <<'PY'
from opsd_research.config import load_config
import sys
print(load_config(sys.argv[1]).data["max_completion_length"])
PY
)"
seed="$(python3 - "${FINOD_CONFIG}" <<'PY'
from opsd_research.config import load_config
import sys
print(load_config(sys.argv[1]).data["seed"])
PY
)"
max_sequence_length="$(python3 - "${FINOD_CONFIG}" <<'PY'
from opsd_research.config import load_config
import sys
print(load_config(sys.argv[1]).data["max_sequence_length"])
PY
)"

if [[ "${FINOD_MAX_STEPS}" == 1 || "${FINOD_MAX_STEPS}" == 5 ]]; then
  export GRAF_SMOKE_MAX_STEPS="${FINOD_MAX_STEPS}"
fi
export GRAF_CONFIG="${PROJECT_SOURCE}/${FINOD_CONFIG}"
export OPSD_DATASET_REVISION=1435fb21d4fecc8ad4966a26f22a874cf2b527f1
export OPSD_EXACT_JSD_VOCAB_CHUNK_SIZE=4096
export OPSD_TAIL_LOGITS_ONLY=1
export OPSD_ADAPTER_STABILITY_INTERVAL=1

vllm_utilization="${FINOD_VLLM_GPU_MEMORY_UTILIZATION:-0.32}"
run_name="finod-g1-128-${FINOD_MAX_STEPS}-${SLURM_JOB_ID}"
output_dir="${TRAIN_ROOT}/${run_name}"
mkdir -p "${output_dir}" logs
cp "${FINOD_CONFIG}" "${output_dir}/config.yaml"
cp "${GRAF_GRAPH_CACHE_MANIFEST}" "${output_dir}/graph-cache-manifest.json"
export OPSD_TRAINING_PARTITION_MANIFEST="${output_dir}/training-partition.json"
printf 'method=finod\nmax_steps=%s\nnum_gpus=8\ngradient_accumulation_steps=4\neffective_batch_size=32\nseed=%s\ndata_seed=%s\nvllm_gpu_memory_utilization=%s\nmax_sequence_length=%s\nmax_grad_norm=0.1\n' \
  "${FINOD_MAX_STEPS}" "${seed}" "${seed}" "${vllm_utilization}" \
  "${max_sequence_length}" > "${output_dir}/runtime-overrides.txt"
record_run_manifest \
  "${output_dir}" runtime-overrides.txt graph-cache-manifest.json

start_gpu_telemetry "${run_name}"
trap stop_gpu_telemetry EXIT

accelerate launch \
  --config_file third_party/opsd/accelerate.yaml \
  --num_processes 8 \
  --gradient_accumulation_steps 4 \
  --main_process_port "${FINOD_PORT}" \
  -m opsd_research.launch_graf_opsd \
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
  --max_steps "${FINOD_MAX_STEPS}" \
  --max_completion_length "${completion_length}" \
  --save_steps "${FINOD_MAX_STEPS}" \
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
  --seed "${seed}" \
  --data_seed "${seed}" \
  --report_to wandb \
  --wandb_project FiNOD-autoresearch

python3 -m opsd_research.summarize_training \
  --training-dir "${output_dir}" \
  --training-log "${PROJECT_SOURCE}/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" \
  --telemetry "${TELEMETRY_ROOT}/${run_name}-${SLURM_JOB_ID}.csv" \
  --output "${output_dir}/training-summary.json" \
  --max-completion-length "${completion_length}"
