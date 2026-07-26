#!/usr/bin/env bash
# Verify G4 before spending the 50-step pilot allocation.
#SBATCH --job-name=graf-after-g4-smoke
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
: "${CACHE_BUILD_JOB_ID:?}" "${G2_VIABILITY_ARRAY_JOB_ID:?}" "${G4_SMOKE_JOB_ID:?}"
training_dir="${TRAIN_ROOT}/graf-g4-viability-long-4096-smoke-${G4_SMOKE_JOB_ID}"
summary="${training_dir}/training-summary.json"
log="${PROJECT_SOURCE}/logs/graf-g4-smoke-${G4_SMOKE_JOB_ID}.out"
test -f "${summary}"
grep -F '"event":"graf_routed_branch_loss_enabled"' "${log}" >/dev/null
python3 - "${log}" "${summary}" <<'PY'
import json, math, sys
active=[]
for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    if '"event":"graf_branch_loss"' in line:
        active.append(float(json.loads(line[line.index("{"):])["active_forks"]))
if not active or max(active) <= 0: raise SystemExit("G4 smoke never activated a branch target")
p=json.load(open(sys.argv[2])); i=p["rollout_dump_integrity"]
if not i["covers_latest_checkpoint"] or i["latest_checkpoint_step"] != 5: raise SystemExit("G4 smoke lacks step-5 rollout")
if any(not math.isfinite(float(x["loss"])) or not math.isfinite(float(x["grad_norm"])) for x in p["loss_history"]): raise SystemExit("G4 smoke is non-finite")
PY
graph_manifest="${SCRATCH_ROOT}/graf_cache/g1-${CACHE_BUILD_JOB_ID}/manifest.json"
viability_manifest="${SCRATCH_ROOT}/graf_cache/g2-viability-${G2_VIABILITY_ARRAY_JOB_ID}/manifest.json"
pilot="$(GRAF_GRAPH_CACHE_MANIFEST="${graph_manifest}" GRAF_VIABILITY_MANIFEST="${viability_manifest}" sbatch --parsable reproductions/06_graf_opsd/g4-pilot.sbatch)"
echo "g4_pilot_job=${pilot}"
sbatch --parsable --dependency="afterok:${pilot}" --export="ALL,CACHE_BUILD_JOB_ID=${CACHE_BUILD_JOB_ID},G2_VIABILITY_ARRAY_JOB_ID=${G2_VIABILITY_ARRAY_JOB_ID},G4_PILOT_JOB_ID=${pilot}" reproductions/06_graf_opsd/after-g4-pilot.sbatch
