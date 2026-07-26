#!/usr/bin/env bash
# Promote the 4,096 control only if the small run is genuinely feasible.
#SBATCH --job-name=graf-after-c1-smoke
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node03
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
module load u22/cuda/12.4
source "${HOME}/OPSD-thinking-research/infra/turing/common.sh"
cd "${PROJECT_SOURCE}"
: "${C1_SMOKE_JOB_ID:?missing C1 smoke job}"
: "${CACHE_BUILD_JOB_ID:?missing graph cache job}" "${G2_VIABILITY_ARRAY_JOB_ID:?missing viability cache job}"
training_dir="${TRAIN_ROOT}/graf-c1-long-4096-smoke-${C1_SMOKE_JOB_ID}"
summary="${training_dir}/training-summary.json"
test -f "${summary}"
python3 - "${summary}" <<'PY'
import json, math, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
integrity = payload["rollout_dump_integrity"]
if not integrity["covers_latest_checkpoint"] or integrity["latest_checkpoint_step"] != 5:
    raise SystemExit("C1 smoke lacks a complete final checkpoint/rollout dump")
for row in payload["loss_history"]:
    if not math.isfinite(float(row["loss"])) or not math.isfinite(float(row["grad_norm"])):
        raise SystemExit("C1 smoke loss or gradient was non-finite")
PY
c1_pilot="$(sbatch --parsable reproductions/06_graf_opsd/c1-pilot.sbatch)"
echo "c1_pilot_job=${c1_pilot}"
sbatch --parsable --dependency="afterok:${c1_pilot}" \
  --export="ALL,C1_PILOT_JOB_ID=${c1_pilot},CACHE_BUILD_JOB_ID=${CACHE_BUILD_JOB_ID},G2_VIABILITY_ARRAY_JOB_ID=${G2_VIABILITY_ARRAY_JOB_ID}" \
  reproductions/06_graf_opsd/after-c1-pilot.sbatch
