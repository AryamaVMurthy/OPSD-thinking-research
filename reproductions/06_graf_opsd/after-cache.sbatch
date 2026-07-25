#!/usr/bin/env bash
#SBATCH --job-name=graf-after-cache
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
manifest="${SCRATCH_ROOT}/graf_cache/g1-${CACHE_BUILD_JOB_ID}/manifest.json"
test -f "${manifest}"
min_accepted="${MIN_GRAPH_ACCEPTED:-64}"
python3 - "${manifest}" "${min_accepted}" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
minimum = int(sys.argv[2])
accepted = int(manifest["accepted_examples"])
if accepted < minimum:
    raise SystemExit(
        f"refusing G1 training: only {accepted} validated graphs; need at least {minimum}"
    )
PY
python3 -m opsd_research.graf_autoresearch register \
  --policy reproductions/06_graf_opsd/configs/autoresearch.yaml \
  --config reproductions/06_graf_opsd/configs/g1-answer-masked-scaffold.yaml \
  --ledger artifacts/graf_opsd/candidate-ledger.jsonl \
  --candidate-id g1-scaffold-2048 \
  --parent-id c0-long-2048 \
  --stage G1 \
  --hypothesis "Answer-masked action graphs reduce raw privileged-teacher leakage." \
  --changed-field graph_mode
g1_job="$(GRAF_GRAPH_CACHE_MANIFEST="${manifest}" sbatch --parsable reproductions/06_graf_opsd/g1-smoke.sbatch)"
echo "g1_smoke_job=${g1_job}"
sbatch --parsable --dependency="afterok:${g1_job}" \
  --export="ALL,CACHE_BUILD_JOB_ID=${CACHE_BUILD_JOB_ID},G1_SMOKE_JOB_ID=${g1_job}" \
  reproductions/06_graf_opsd/after-g1-smoke.sbatch
