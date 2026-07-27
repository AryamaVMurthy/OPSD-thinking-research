#!/usr/bin/env bash
# Four independent Qwen3-4B replicas maximize throughput for the 128-row cache.
#SBATCH --job-name=ch-cache
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node03
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=350G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module load u22/cuda/12.4
source "${HOME}/OPSD-thinking-research/infra/turing/common.sh"
cd "${PROJECT_SOURCE}"

CH_LIMIT="${CH_LIMIT:-128}"
CH_BLIND_ATTEMPTS="${CH_BLIND_ATTEMPTS:-3}"
CH_AUDIT_ATTEMPTS="${CH_AUDIT_ATTEMPTS:-3}"
CH_BLIND_MAX_TOKENS="${CH_BLIND_MAX_TOKENS:-2048}"
CH_AUDIT_MAX_TOKENS="${CH_AUDIT_MAX_TOKENS:-1536}"
CH_MAX_MODEL_LEN="${CH_MAX_MODEL_LEN:-28672}"
if (( CH_LIMIT < 1 || CH_AUDIT_ATTEMPTS < 1 || CH_BLIND_MAX_TOKENS < 1 ||
      CH_AUDIT_MAX_TOKENS < 1 || CH_BLIND_MAX_TOKENS >= CH_MAX_MODEL_LEN ||
      CH_AUDIT_MAX_TOKENS >= CH_MAX_MODEL_LEN )); then
  echo "CH example, attempt, generation, and context limits are invalid" >&2
  exit 2
fi
if [[ "${CH_BLIND_ATTEMPTS}" != 2 && "${CH_BLIND_ATTEMPTS}" != 3 ]]; then
  echo "CH_BLIND_ATTEMPTS must be 2 or 3" >&2
  exit 2
fi

CH_RUN_DIR="${SCRATCH_ROOT}/ch_cache/ch-${CH_LIMIT}-${SLURM_JOB_ID}"
export CH_RUN_DIR CH_LIMIT CH_BLIND_ATTEMPTS CH_AUDIT_ATTEMPTS
export CH_BLIND_MAX_TOKENS CH_AUDIT_MAX_TOKENS CH_MAX_MODEL_LEN
mkdir -p "${CH_RUN_DIR}"
git rev-parse HEAD > "${CH_RUN_DIR}/source-commit.txt"
{
  printf 'limit=%s\n' "${CH_LIMIT}"
  printf 'blind_attempts=%s\n' "${CH_BLIND_ATTEMPTS}"
  printf 'audit_attempts=%s\n' "${CH_AUDIT_ATTEMPTS}"
  printf 'num_shards=%s\n' "${SLURM_NTASKS}"
  printf 'blind_max_tokens=%s\n' "${CH_BLIND_MAX_TOKENS}"
  printf 'audit_max_tokens=%s\n' "${CH_AUDIT_MAX_TOKENS}"
  printf 'max_model_len=%s\n' "${CH_MAX_MODEL_LEN}"
  printf 'parallelism=%s-independent-vllm-replicas\n' "${SLURM_NTASKS}"
} > "${CH_RUN_DIR}/runtime-parameters.txt"

start_gpu_telemetry "ch-cache-${CH_LIMIT}"
trap stop_gpu_telemetry EXIT
srun \
  --kill-on-bad-exit=1 \
  --gpus-per-task=1 \
  --gpu-bind=single:1 \
  "${PROJECT_SOURCE}/infra/turing/ch_cache_worker.sh"

shards=()
for ((shard_id = 0; shard_id < SLURM_NTASKS; shard_id++)); do
  shards+=("${CH_RUN_DIR}/dossiers.shard${shard_id}.jsonl")
done
python3 -m opsd_research.merge_ch_cache \
  --shards "${shards[@]}" \
  --output "${CH_RUN_DIR}/dossiers.jsonl" \
  --manifest "${CH_RUN_DIR}/manifest.json" \
  --requested-examples "${CH_LIMIT}" \
  --blind-attempts-per-problem "${CH_BLIND_ATTEMPTS}" \
  --builder-model Qwen/Qwen3-4B \
  --builder-model-revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --training-dataset-revision 1435fb21d4fecc8ad4966a26f22a874cf2b527f1 \
  --builder-seed 42

python3 -m opsd_research.summarize_ch_cache \
  --input "${CH_RUN_DIR}/dossiers.jsonl" \
  --output "${CH_RUN_DIR}/cache-summary.json"

sha256sum \
  "${CH_RUN_DIR}"/dossiers.shard*.jsonl \
  "${CH_RUN_DIR}/dossiers.jsonl" \
  "${CH_RUN_DIR}/manifest.json" \
  "${CH_RUN_DIR}/cache-summary.json" \
  "${CH_RUN_DIR}/runtime-parameters.txt" \
  "${CH_RUN_DIR}/source-commit.txt" \
  > "${CH_RUN_DIR}/artifact-manifest.sha256"
printf 'CH_DOSSIER_MANIFEST=%s\n' "${CH_RUN_DIR}/manifest.json"
