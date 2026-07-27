#!/usr/bin/env bash
# Launch one diversity-scaled confirmation after the neutral CH1 pilot.
#SBATCH --job-name=submit-ch1-diversity
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
python3 - "${treatment}/paired-vs-base.json" "${treatment}/summary.json" <<'PY'
import json, sys

comparison = json.load(open(sys.argv[1], encoding="utf-8"))
summary = json.load(open(sys.argv[2], encoding="utf-8"))
delta = float(comparison["avg_at_4"]["delta"])
if summary.get("method") != "ch1-completion-aware-4096":
    raise SystemExit("confirmation input is not the pinned CH1 method")
if summary.get("checkpoint") != "step-12":
    raise SystemExit("confirmation input is not the completed 12-step pilot")
if not comparison.get("pairing_verified") or int(
    comparison.get("samples_per_problem", 0)
) != 4:
    raise SystemExit("confirmation requires the paired four-sample protocol")
if delta != 0.0:
    raise SystemExit(
        f"diversity confirmation is reserved for neutral CH1; delta={delta:+.6f}"
    )
if float(summary.get("length_cutoff_rate", 1.0)) > 0.05:
    raise SystemExit("confirmation input is confounded by evaluation truncation")
print(json.dumps({
    "event": "ch1_diversity_confirmation_gate",
    "pilot_avg_at_4_delta": delta,
    "decision": "launch-one-diversity-scaled-confirmation",
}, sort_keys=True))
PY

confirmation_scratch="/scratch/node10/${USER}/opsd-thinking-research"
cache="$(
  sbatch --parsable \
    --nodelist=node10 \
    --gres=gpu:8 \
    --ntasks=8 \
    --cpus-per-task=8 \
    --mem=900G \
    --time=03:00:00 \
    --output="${confirmation_scratch}/slurm-ch-cache-%j.out" \
    --export=ALL,CH_LIMIT=1682,CH_BLIND_ATTEMPTS=2,CH_AUDIT_ATTEMPTS=3,CH_BLIND_MAX_TOKENS=4096,CH_AUDIT_MAX_TOKENS=1536,CH_MAX_MODEL_LEN=28672 \
    reproductions/06_graf_opsd/build-ch-cache.sbatch
)"
echo "ch1_diversity_cache_job=${cache}"
controller="$(
  sbatch --parsable \
    --dependency="afterok:${cache}" \
    --output="${confirmation_scratch}/slurm-after-ch-diversity-cache-%j.out" \
    --export="ALL,CH_CONFIRM_CACHE_JOB_ID=${cache},CH_CONFIRM_LIMIT=1682" \
    reproductions/06_graf_opsd/after-ch-diversity-cache.sbatch
)"
echo "ch1_diversity_cache_controller=${controller}"
