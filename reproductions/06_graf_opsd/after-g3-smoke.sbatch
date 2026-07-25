#!/usr/bin/env bash
# Validate G3's entropy-preserving objective before its AIME24 development pilot.
#SBATCH --job-name=graf-after-g3-smoke
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
: "${G2_VIABILITY_ARRAY_JOB_ID:?missing viability cache job}"
: "${G3_SMOKE_JOB_ID:?missing G3 smoke job}"
training_dir="${TRAIN_ROOT}/graf-g3-viability-entropy-smoke-${G3_SMOKE_JOB_ID}"
summary="${training_dir}/training-summary.json"
log="${PROJECT_SOURCE}/logs/graf-g3-smoke-${G3_SMOKE_JOB_ID}.out"
test -f "${summary}"
grep -F '"event":"graf_routed_branch_loss_enabled"' "${log}" >/dev/null
grep -F '"event":"graf_branch_loss"' "${log}" >/dev/null
python3 - "${summary}" <<'PY'
import json, math, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
integrity = payload["rollout_dump_integrity"]
if not integrity["covers_latest_checkpoint"] or integrity["latest_checkpoint_step"] != 5:
    raise SystemExit("G3 smoke lacks a complete final checkpoint/rollout dump")
for row in payload["loss_history"]:
    if not math.isfinite(float(row["loss"])) or not math.isfinite(float(row["grad_norm"])):
        raise SystemExit("G3 smoke loss or gradient was non-finite")
PY
graph_manifest="${SCRATCH_ROOT}/graf_cache/g1-${CACHE_BUILD_JOB_ID}/manifest.json"
viability_manifest="${SCRATCH_ROOT}/graf_cache/g2-viability-${G2_VIABILITY_ARRAY_JOB_ID}/manifest.json"
g3_pilot="$(GRAF_GRAPH_CACHE_MANIFEST="${graph_manifest}" GRAF_VIABILITY_MANIFEST="${viability_manifest}" sbatch --parsable reproductions/06_graf_opsd/g3-pilot.sbatch)"
echo "g3_pilot_job=${g3_pilot}"
sbatch --parsable --dependency="afterok:${g3_pilot}" \
  --export="ALL,G3_PILOT_JOB_ID=${g3_pilot}" \
  reproductions/06_graf_opsd/after-g3-pilot.sbatch
