#!/usr/bin/env bash
# Verify privacy plumbing, finite optimization, and complete smoke rollouts.
#SBATCH --job-name=after-ch-smoke
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
: "${CH_CACHE_JOB_ID:?}" "${CH_LIMIT:?}" "${CH_SMOKE_JOB_ID:?}"
: "${CH_CANDIDATE_CONFIG:?}" "${CH_CANDIDATE_ID:?}" "${CH_PILOT_STEPS:?}"
manifest="${SCRATCH_ROOT}/ch_cache/ch-${CH_LIMIT}-${CH_CACHE_JOB_ID}/manifest.json"
run_name="ch-${CH_CANDIDATE_ID}-5-${CH_SMOKE_JOB_ID}"
training_dir="${TRAIN_ROOT}/${run_name}"
summary="${training_dir}/training-summary.json"
log="${PROJECT_SOURCE}/logs/ch-opsd-train-${CH_SMOKE_JOB_ID}.out"
test -f "${summary}"
test -f "${training_dir}/adapter_model.safetensors"
grep -F '"event":"ch_teacher_only_dossier_enabled"' "${log}" >/dev/null
grep -F '"student_answer_context":false' "${log}" >/dev/null
python3 - "${summary}" <<'PY'
import json, math, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
i = p["rollout_dump_integrity"]
if not i["covers_latest_checkpoint"] or i["latest_checkpoint_step"] != 5:
    raise SystemExit("CH smoke lacks its step-5 rollout")
if any(not math.isfinite(float(row["loss"])) or
       not math.isfinite(float(row["grad_norm"])) for row in p["loss_history"]):
    raise SystemExit("CH smoke optimization is non-finite")
PY

pilot="$(
  CH_DOSSIER_MANIFEST="${manifest}" \
  CH_CANDIDATE_CONFIG="${CH_CANDIDATE_CONFIG}" \
  CH_CANDIDATE_ID="${CH_CANDIDATE_ID}" \
  CH_MAX_STEPS="${CH_PILOT_STEPS}" \
  CH_PORT=12986 \
  sbatch --parsable reproductions/06_graf_opsd/run-ch-candidate.sbatch
)"
echo "ch_pilot_job=${pilot}"
sbatch \
  --parsable \
  --dependency="afterok:${pilot}" \
  --export="ALL,CH_CACHE_JOB_ID=${CH_CACHE_JOB_ID},CH_LIMIT=${CH_LIMIT},CH_PILOT_JOB_ID=${pilot},CH_CANDIDATE_ID=${CH_CANDIDATE_ID},CH_PILOT_STEPS=${CH_PILOT_STEPS}" \
  reproductions/06_graf_opsd/after-ch-pilot.sbatch
