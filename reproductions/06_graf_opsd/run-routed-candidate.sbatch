#!/usr/bin/env bash
# Generic audited runner for a GRAF routed smoke or development pilot.
#SBATCH --job-name=graf-routed
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node03
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=350G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
module load u22/cuda/12.4
source "${HOME}/OPSD-thinking-research/infra/turing/common.sh"
cd "${PROJECT_SOURCE}"
: "${GRAF_CANDIDATE_ID:?}" "${GRAF_CANDIDATE_CONFIG:?}" "${GRAF_MAX_STEPS:?}" "${GRAF_PORT:?}"
python3 -m opsd_research.cli validate-config "${GRAF_CANDIDATE_CONFIG}"
completion_length="$(python3 - "${GRAF_CANDIDATE_CONFIG}" <<'PY'
from opsd_research.config import load_config
import sys
print(load_config(sys.argv[1]).data["max_completion_length"])
PY
)"
token_divergence="$(python3 - "${GRAF_CANDIDATE_CONFIG}" <<'PY'
from opsd_research.config import load_config
import sys
print(load_config(sys.argv[1]).data.get("token_divergence", "legacy_clipped_forward_kl"))
PY
)"
graph_mode="$(python3 - "${GRAF_CANDIDATE_CONFIG}" <<'PY'
from opsd_research.config import load_config
import sys
print(load_config(sys.argv[1]).data["graph_mode"])
PY
)"
max_sequence_length="$(python3 - "${GRAF_CANDIDATE_CONFIG}" <<'PY'
from opsd_research.config import load_config
import sys
print(load_config(sys.argv[1]).data.get("max_sequence_length", 20000))
PY
)"
graph_modes_requiring_graph=(
  "scaffold_graph" "fork_mask" "viability_routed"
  "fluid_viability_routed" "fluid_context_dossier"
)
graph_modes_requiring_viability=(
  "viability_routed" "fluid_viability_routed" "fluid_context_dossier"
)
graph_modes_requiring_dossier=(
  "context_dossier" "fluid_context_dossier" "fluid_viability_routed"
)
requires_graph=0
requires_viability=0
requires_dossier=0
for mode in "${graph_modes_requiring_graph[@]}"; do
  [[ "${graph_mode}" == "${mode}" ]] && requires_graph=1
done
for mode in "${graph_modes_requiring_viability[@]}"; do
  [[ "${graph_mode}" == "${mode}" ]] && requires_viability=1
done
for mode in "${graph_modes_requiring_dossier[@]}"; do
  [[ "${graph_mode}" == "${mode}" ]] && requires_dossier=1
done
if [[ "${requires_graph}" == 1 ]]; then
  : "${GRAF_GRAPH_CACHE_MANIFEST:?graph mode requires GRAF_GRAPH_CACHE_MANIFEST}"
fi
if [[ "${requires_viability}" == 1 ]]; then
  : "${GRAF_VIABILITY_MANIFEST:?graph mode requires GRAF_VIABILITY_MANIFEST}"
fi
if [[ "${requires_dossier}" == 1 ]]; then
  : "${CH_DOSSIER_MANIFEST:?teacher context requires CH_DOSSIER_MANIFEST}"
fi
if [[ "${GRAF_MAX_STEPS}" == 1 || "${GRAF_MAX_STEPS}" == 5 ]]; then
  export GRAF_SMOKE_MAX_STEPS="${GRAF_MAX_STEPS}"
fi
export GRAF_CONFIG="${PROJECT_SOURCE}/${GRAF_CANDIDATE_CONFIG}" OPSD_DATASET_REVISION=1435fb21d4fecc8ad4966a26f22a874cf2b527f1 OPSD_EXACT_JSD_VOCAB_CHUNK_SIZE=4096 OPSD_TAIL_LOGITS_ONLY=1
if [[ "${token_divergence}" == "legacy_clipped_forward_kl" ]]; then
  jsd_token_clip=0.05
  export OPSD_DIVERGENCE_DIAGNOSTICS_INTERVAL=1
else
  jsd_token_clip=0
  # A one-step smoke must produce a diagnostic snapshot. Longer pilots sample
  # one microbatch in eight to keep full-vocabulary telemetry inexpensive.
  if [[ "${GRAF_MAX_STEPS}" == 1 ]]; then
    export OPSD_DIVERGENCE_DIAGNOSTICS_INTERVAL=1
  else
    export OPSD_DIVERGENCE_DIAGNOSTICS_INTERVAL=8
  fi
fi
# Keep candidate runs within the memory envelope verified by the 4,096-token
# G4 pilot.  Callers may still raise this deliberately after a measured memory
# audit, but an implicit larger reservation should not turn an autoresearch
# branch into an unrelated OOM experiment.
vllm_utilization="${GRAF_VLLM_GPU_MEMORY_UTILIZATION:-0.35}"
run_name="graf-${GRAF_CANDIDATE_ID}-${GRAF_MAX_STEPS}-${SLURM_JOB_ID}"; output_dir="${TRAIN_ROOT}/${run_name}"
mkdir -p "${output_dir}"
cp "${GRAF_CANDIDATE_CONFIG}" "${output_dir}/config.yaml"
manifest_inputs=(runtime-overrides.txt)
if [[ "${requires_graph}" == 1 ]]; then
  cp "${GRAF_GRAPH_CACHE_MANIFEST}" "${output_dir}/graph-cache-manifest.json"
  manifest_inputs+=(graph-cache-manifest.json)
fi
if [[ "${requires_viability}" == 1 ]]; then
  cp "${GRAF_VIABILITY_MANIFEST}" "${output_dir}/viability-manifest.json"
  manifest_inputs+=(viability-manifest.json)
fi
if [[ "${requires_dossier}" == 1 ]]; then
  cp "${CH_DOSSIER_MANIFEST}" "${output_dir}/ch-dossier-manifest.json"
  manifest_inputs+=(ch-dossier-manifest.json)
fi
export OPSD_TRAINING_PARTITION_MANIFEST="${output_dir}/training-partition.json"
export OPSD_ADAPTER_STABILITY_INTERVAL=1
export OPSD_RECOMPUTE_DIVERGENCE_BACKWARD=1
printf 'candidate=%s\nmax_steps=%s\nnum_gpus=4\ngradient_accumulation_steps=8\neffective_batch_size=32\nvllm_gpu_memory_utilization=%s\ntoken_divergence=%s\ngraph_mode=%s\nmax_sequence_length=%s\njsd_token_clip=%s\ndivergence_diagnostics_interval=%s\nadapter_stability_interval=%s\nrecomputed_divergence_backward=%s\n' "${GRAF_CANDIDATE_ID}" "${GRAF_MAX_STEPS}" "${vllm_utilization}" "${token_divergence}" "${graph_mode}" "${max_sequence_length}" "${jsd_token_clip}" "${OPSD_DIVERGENCE_DIAGNOSTICS_INTERVAL}" "${OPSD_ADAPTER_STABILITY_INTERVAL}" "${OPSD_RECOMPUTE_DIVERGENCE_BACKWARD}" > "${output_dir}/runtime-overrides.txt"
record_run_manifest "${output_dir}" "${manifest_inputs[@]}"
start_gpu_telemetry "${run_name}"; trap stop_gpu_telemetry EXIT
accelerate launch --config_file third_party/opsd/accelerate.yaml --num_processes 4 --gradient_accumulation_steps 8 --main_process_port "${GRAF_PORT}" \
  -m opsd_research.launch_graf_opsd --model_name_or_path Qwen/Qwen3-4B --model_revision 1cfa9a7208912126459214e8b04321603b3df60c --student_model_revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --learning_rate 5e-6 --max_grad_norm 0.1 --per_device_train_batch_size 1 --gradient_checkpointing --gradient_accumulation_steps 8 --output_dir "${TRAIN_ROOT}" --run_config "${run_name}" --num_train_epochs 30 --max_steps "${GRAF_MAX_STEPS}" --max_completion_length "${completion_length}" --save_steps "${GRAF_MAX_STEPS}" --logging_steps 1 \
  --attn_implementation flash_attention_2 --torch_dtype bfloat16 --max_length "${max_sequence_length}" --beta 0 --use_vllm --vllm_mode colocate --vllm_gpu_memory_utilization "${vllm_utilization}" --vllm_tensor_parallel_size 1 --use_peft --lora_r 64 --lora_alpha 128 --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj --temperature 1.1 --top_p 0.95 --top_k 20 --lmbda 1 --fixed_teacher --student_thinking --teacher_thinking --jsd_token_clip "${jsd_token_clip}" --seed 42 --data_seed 42 --report_to wandb --wandb_project GRAF-OPSD-autoresearch
python3 -m opsd_research.summarize_training --training-dir "${output_dir}" --training-log "${PROJECT_SOURCE}/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" --telemetry "${TELEMETRY_ROOT}/${run_name}-${SLURM_JOB_ID}.csv" --output "${output_dir}/training-summary.json" --max-completion-length "${completion_length}"
