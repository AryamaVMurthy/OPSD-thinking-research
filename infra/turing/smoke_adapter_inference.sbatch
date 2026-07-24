#!/usr/bin/env bash
#SBATCH --job-name=opsd-adapter-smoke
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node10
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module load u22/cuda/12.4
source "${HOME}/OPSD-thinking-research/infra/turing/common.sh"
cd "${PROJECT_SOURCE}"

: "${CONFIG:?set CONFIG to a math-evaluation YAML path}"
: "${RUN_NAME:?set RUN_NAME}"
: "${ADAPTER:?set ADAPTER to a LoRA checkpoint directory}"
: "${CHECKPOINT:?set CHECKPOINT}"

python3 -m opsd_research.cli validate-config "${CONFIG}"
adapter_file="${ADAPTER}/adapter_model.safetensors"
if [[ ! -f "${adapter_file}" ]]; then
  echo "Missing LoRA adapter: ${adapter_file}" >&2
  exit 1
fi
adapter_sha256="$(sha256sum "${adapter_file}" | awk '{print $1}')"

output_dir="${RESULTS_ROOT}/${RUN_NAME}"
output="${output_dir}/generations.jsonl"
mkdir -p "${output_dir}"
cp "${CONFIG}" "${output_dir}/config.yaml"
{
  printf 'adapter_path=%s\n' "${ADAPTER}"
  printf 'adapter_sha256=%s\n' "${adapter_sha256}"
  printf 'method=opsd-standard-thinking-smoke\n'
  printf 'checkpoint=%s\n' "${CHECKPOINT}"
} > "${output_dir}/adapter-metadata.txt"
record_run_manifest "${output_dir}" adapter-metadata.txt

start_gpu_telemetry "${RUN_NAME}"
trap stop_gpu_telemetry EXIT

python3 -m opsd_research.math_eval \
  --config "${CONFIG}" \
  --output "${output}" \
  --adapter "${ADAPTER}" \
  --adapter-sha256 "${adapter_sha256}" \
  --method opsd-standard-thinking-smoke \
  --checkpoint "${CHECKPOINT}" \
  --limit 1
python3 -m opsd_research.verify_smoke --input "${output}"
