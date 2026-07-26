#!/usr/bin/env bash
# Explicit promotion gate from the bounded paired CH1 screen to paper-scale work.
# This controller runs on node03 because the pilot comparison is node-local.
#SBATCH --job-name=promote-ch1-full
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
: "${CH_PILOT_EVAL_RUN:?}"

treatment="${RESULTS_ROOT}/${CH_PILOT_EVAL_RUN}"
comparison="${treatment}/paired-vs-base.json"
summary="${treatment}/summary.json"
test -f "${comparison}"
test -f "${summary}"
python3 - "${comparison}" "${summary}" <<'PY'
import json, sys
comparison = json.load(open(sys.argv[1], encoding="utf-8"))
summary = json.load(open(sys.argv[2], encoding="utf-8"))
delta = float(comparison["avg_at_4"]["delta"])
if summary.get("method") != "ch1-completion-aware-4096":
    raise SystemExit("promotion input is not the pinned CH1 pilot")
if delta <= 0:
    raise SystemExit(
        f"CH1 paired avg@4 delta {delta:+.6f} is not positive; "
        "paper-scale work is not authorized by the evidence gate"
    )
print(json.dumps({
    "event": "ch1_full_promotion_gate",
    "paired_avg_at_4_delta": delta,
    "decision": "launch-paper-scale-confirmation",
}, sort_keys=True))
PY

full_scratch="/scratch/node10/${USER}/opsd-thinking-research"
cache="$(
  sbatch --parsable \
    --nodelist=node10 \
    --gres=gpu:8 \
    --ntasks=8 \
    --cpus-per-task=8 \
    --mem=900G \
    --time=06:00:00 \
    --output="${full_scratch}/slurm-ch-cache-%j.out" \
    --export=ALL,CH_LIMIT=6912,CH_BLIND_ATTEMPTS=2,CH_AUDIT_ATTEMPTS=3,CH_BLIND_MAX_TOKENS=4096,CH_AUDIT_MAX_TOKENS=1536,CH_MAX_MODEL_LEN=28672 \
    reproductions/06_graf_opsd/build-ch-cache.sbatch
)"
echo "ch1_full_cache_job=${cache}"
controller="$(
  sbatch --parsable \
    --dependency="afterok:${cache}" \
    --output="${full_scratch}/slurm-after-ch-full-cache-%j.out" \
    --export="ALL,CH_FULL_CACHE_JOB_ID=${cache},CH_FULL_LIMIT=6912" \
    reproductions/06_graf_opsd/after-ch-full-cache.sbatch
)"
echo "ch1_full_cache_controller=${controller}"
