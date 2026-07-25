#!/usr/bin/env bash
# Validate G2's branch objective before spending a development pilot.
#SBATCH --job-name=graf-after-g2-smoke
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

: "${CACHE_BUILD_JOB_ID:?missing graph cache job}"
: "${G2_VIABILITY_ARRAY_JOB_ID:?missing viability array job}"
: "${G2_SMOKE_JOB_ID:?missing G2 smoke job}"
training_dir="${TRAIN_ROOT}/graf-g2-viability-smoke-${G2_SMOKE_JOB_ID}"
summary="${training_dir}/training-summary.json"
log="${PROJECT_SOURCE}/logs/graf-g2-smoke-${G2_SMOKE_JOB_ID}.out"
test -f "${summary}"
grep -F '"event":"graf_routed_branch_loss_enabled"' "${log}" >/dev/null
grep -F '"event":"graf_branch_loss"' "${log}" >/dev/null
python3 - "${summary}" <<'PY'
import json
import math
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
integrity = payload["rollout_dump_integrity"]
if not integrity["covers_latest_checkpoint"] or integrity["latest_checkpoint_step"] != 5:
    raise SystemExit("G2 smoke lacks a complete final checkpoint/rollout dump")
for row in payload["loss_history"]:
    if not math.isfinite(float(row["loss"])) or not math.isfinite(float(row["grad_norm"])):
        raise SystemExit("G2 smoke loss or gradient was non-finite")
PY
graph_manifest="${SCRATCH_ROOT}/graf_cache/g1-${CACHE_BUILD_JOB_ID}/manifest.json"
viability_manifest="${SCRATCH_ROOT}/graf_cache/g2-viability-${G2_VIABILITY_ARRAY_JOB_ID}/manifest.json"
g2_pilot="$(GRAF_GRAPH_CACHE_MANIFEST="${graph_manifest}" GRAF_VIABILITY_MANIFEST="${viability_manifest}" sbatch --parsable reproductions/06_graf_opsd/g2-pilot.sbatch)"
echo "g2_pilot_job=${g2_pilot}"
sbatch --parsable --dependency="afterok:${g2_pilot}" \
  --export="ALL,G2_PILOT_JOB_ID=${g2_pilot}" \
  reproductions/06_graf_opsd/after-g2-pilot.sbatch
