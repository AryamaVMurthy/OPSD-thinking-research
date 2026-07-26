#!/usr/bin/env bash
# Admit only a sufficiently complete immutable CH cache, then launch smoke.
#SBATCH --job-name=after-ch-cache
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
: "${CH_CACHE_JOB_ID:?}" "${CH_LIMIT:?}"
manifest="${SCRATCH_ROOT}/ch_cache/ch-${CH_LIMIT}-${CH_CACHE_JOB_ID}/manifest.json"
summary="${SCRATCH_ROOT}/ch_cache/ch-${CH_LIMIT}-${CH_CACHE_JOB_ID}/cache-summary.json"
test -f "${manifest}"
test -f "${summary}"
python3 - "${manifest}" "${summary}" <<'PY'
import json, sys
from opsd_research.ch_dossier import accepted_teacher_dossiers

path = sys.argv[1]
payload = json.load(open(path, encoding="utf-8"))
summary = json.load(open(sys.argv[2], encoding="utf-8"))
accepted = accepted_teacher_dossiers(path)
rate = len(accepted) / int(payload["requested_examples"])
print(json.dumps({"event": "ch_cache_gate", "accepted": len(accepted),
                  "requested": payload["requested_examples"],
                  "acceptance_rate": rate,
                  "third_attempt_unique_recovery_rate":
                      summary["third_attempt_unique_recovery_rate"],
                  "total_generation_tokens":
                      summary["total_generation_tokens"]}, sort_keys=True))
if rate < 0.95:
    raise SystemExit("CH cache acceptance rate is below the 95% coverage gate")
PY

smoke="$(
  CH_DOSSIER_MANIFEST="${manifest}" \
  CH_CANDIDATE_CONFIG=reproductions/06_graf_opsd/configs/ch0-contrastive-hindsight-2048.yaml \
  CH_CANDIDATE_ID=ch0-contrastive-hindsight-2048 \
  CH_MAX_STEPS=5 \
  CH_PORT=12985 \
  sbatch --parsable reproductions/06_graf_opsd/run-ch-candidate.sbatch
)"
echo "ch_smoke_job=${smoke}"
sbatch \
  --parsable \
  --dependency="afterok:${smoke}" \
  --export="ALL,CH_CACHE_JOB_ID=${CH_CACHE_JOB_ID},CH_LIMIT=${CH_LIMIT},CH_SMOKE_JOB_ID=${smoke}" \
  reproductions/06_graf_opsd/after-ch-smoke.sbatch
