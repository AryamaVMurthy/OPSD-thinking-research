#!/usr/bin/env bash
#SBATCH --job-name=graf-after-g1-smoke
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node10
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module load u22/cuda/12.4
source "${HOME}/OPSD-thinking-research/infra/turing/common.sh"
cd "${PROJECT_SOURCE}"

: "${CACHE_BUILD_JOB_ID:?missing graph-cache build job ID}"
: "${G1_SMOKE_JOB_ID:?missing G1 smoke job ID}"
smoke_dir="${TRAIN_ROOT}/graf-g1-answer-masked-smoke-${G1_SMOKE_JOB_ID}"
summary="${smoke_dir}/training-summary.json"
log="${PROJECT_SOURCE}/logs/graf-g1-smoke-${G1_SMOKE_JOB_ID}.out"
test -f "${summary}"
grep -F '"event":"graf_answer_masked_scaffold_enabled"' "${log}" >/dev/null
python3 - "${summary}" <<'PY'
import json
import math
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload["rollout_dump_integrity"] != {
    "covers_latest_checkpoint": True,
    "latest_checkpoint_step": 5,
    "latest_generation_step": 5,
}:
    raise SystemExit("G1 smoke did not save a complete step-5 checkpoint/rollout pair")
for row in payload["loss_history"]:
    if not math.isfinite(float(row["loss"])) or not math.isfinite(float(row["grad_norm"])):
        raise SystemExit("G1 smoke loss or gradient was non-finite")
PY
manifest="${SCRATCH_ROOT}/graf_cache/g1-${CACHE_BUILD_JOB_ID}/manifest.json"
g1_pilot="$(GRAF_GRAPH_CACHE_MANIFEST="${manifest}" sbatch --parsable reproductions/06_graf_opsd/g1-pilot.sbatch)"
echo "g1_pilot_job=${g1_pilot}"
sbatch --parsable --dependency="afterok:${g1_pilot}" \
  --export="ALL,CACHE_BUILD_JOB_ID=${CACHE_BUILD_JOB_ID},G1_PILOT_JOB_ID=${g1_pilot}" \
  reproductions/06_graf_opsd/after-g1-pilot.sbatch
