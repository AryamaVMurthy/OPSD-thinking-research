#!/usr/bin/env bash
# Build independent problem-only plans on all eight node10 GPUs.
#SBATCH --job-name=fisher-guidance
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node10
#SBATCH --gres=gpu:8
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=900G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail

module load u22/cuda/12.4
: "${PROJECT_SOURCE:?}"
source "${HOME}/OPSD-thinking-research/infra/turing/common.sh"
cd "${PROJECT_SOURCE}"
test -z "$(git status --short)"

export FISHER_GUIDANCE_LIMIT="${FISHER_GUIDANCE_LIMIT:-512}"
export FISHER_GUIDANCE_MIN_ACCEPTED="${FISHER_GUIDANCE_MIN_ACCEPTED:-384}"
export FISHER_GUIDANCE_SHARDS=8
export FISHER_GUIDANCE_ROOT="${FISHER_GUIDANCE_ROOT:-${SCRATCH_ROOT}/fisher-guidance/contest-${SLURM_JOB_ID}}"
mkdir -p "${FISHER_GUIDANCE_ROOT}"
git rev-parse HEAD > "${FISHER_GUIDANCE_ROOT}/source-commit.txt"

start_gpu_telemetry "fisher-guidance-${SLURM_JOB_ID}"
trap stop_gpu_telemetry EXIT
srun \
  --kill-on-bad-exit=1 \
  --gpus-per-task=1 \
  --gpu-bind=single:1 \
  "${PROJECT_SOURCE}/infra/turing/build_fisher_guidance_worker.sh"

manifests=()
for shard in 0 1 2 3 4 5 6 7; do
  manifests+=("${FISHER_GUIDANCE_ROOT}/shard-${shard}/manifest.json")
done
python3 -m opsd_research.merge_fisher_guidance \
  --manifest "${manifests[@]}" \
  --output "${FISHER_GUIDANCE_ROOT}/plans.jsonl" \
  --manifest-output "${FISHER_GUIDANCE_ROOT}/manifest.json"
python3 - "${FISHER_GUIDANCE_ROOT}/manifest.json" "${FISHER_GUIDANCE_MIN_ACCEPTED}" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
if int(manifest["accepted_records"]) < int(sys.argv[2]):
    raise SystemExit(
        f"guidance accepted {manifest['accepted_records']}, "
        f"fewer than required {sys.argv[2]}"
    )
PY
sha256sum \
  "${FISHER_GUIDANCE_ROOT}/plans.jsonl" \
  "${FISHER_GUIDANCE_ROOT}/manifest.json" \
  "${FISHER_GUIDANCE_ROOT}/source-commit.txt" \
  > "${FISHER_GUIDANCE_ROOT}/artifact-manifest.sha256"
printf 'FISHER_GUIDANCE_MANIFEST=%s\n' \
  "${FISHER_GUIDANCE_ROOT}/manifest.json"
