#!/usr/bin/env bash
# Admit a clean one-step FiNOD smoke and submit the five-step signal screen.
#SBATCH --job-name=admit-finod
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node10
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail

: "${PROJECT_SOURCE:?}"
: "${FINOD_SMOKE_RUN_NAME:?}"
: "${FINOD_CACHE_JOB_ID:?}"
source "${HOME}/OPSD-thinking-research/infra/turing/common.sh"
cd "${PROJECT_SOURCE}"
summary="${TRAIN_ROOT}/${FINOD_SMOKE_RUN_NAME}/training-summary.json"
manifest="${
  SCRATCH_ROOT
}/finod/cache/representative-${FINOD_CACHE_JOB_ID}/manifest.json"
test -s "${summary}"
test -s "${manifest}"

python3 - "${summary}" <<'PY'
import json
import math
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
signal = summary["finod_signal"]
history = summary["finod_signal_history"]
if not signal["all_finite"] or not summary["loss_finite"]:
    raise SystemExit("FiNOD smoke has a non-finite loss or signal")
if int(signal["events"]) < 1:
    raise SystemExit("FiNOD smoke recorded no unrounded signal events")
if int(signal["positive_loss_events"]) != int(signal["events"]):
    raise SystemExit("FiNOD smoke contains a zero/nonpositive loss event")
if not float(signal["mean_residual_energy"]) > 1e-8:
    raise SystemExit("FiNOD smoke residual Fisher energy collapsed")
if not float(signal["mean_collapsed_residual_fraction"]) < 0.8:
    raise SystemExit("FiNOD smoke collapsed at least 80% of selected tokens")
if float(signal["max_target_kl"]) > 0.010001:
    raise SystemExit("FiNOD smoke exceeded the registered target-KL cap")
if any(
    float(event["positive_alignment_after_fraction"]) > 1e-7
    for event in history
):
    raise SystemExit("FiNOD smoke retained positive nuisance alignment")
losses = summary["loss_history"]
if not losses or any(
    not math.isfinite(float(item["grad_norm"])) for item in losses
):
    raise SystemExit("FiNOD smoke lacks finite optimizer gradients")
if not summary["checkpoints"] or not summary["checkpoints"][-1]["adapter_present"]:
    raise SystemExit("FiNOD smoke did not save a nonempty adapter checkpoint")
if not summary["rollout_dump_integrity"]["covers_latest_checkpoint"]:
    raise SystemExit("FiNOD smoke rollout evidence misses the latest checkpoint")
PY

five_job="$(
  sbatch --parsable \
    --export="ALL,PROJECT_SOURCE=${PROJECT_SOURCE},FINOD_CONFIG=reproductions/06_graf_opsd/configs/finod-g1-representative-1024.yaml,FINOD_MAX_STEPS=5,FINOD_PORT=12992,FINOD_RUN_NAME=finod-problem-only-1024-smoke5,GRAF_GRAPH_CACHE_MANIFEST=${manifest}" \
    reproductions/06_graf_opsd/run-finod-candidate.sbatch
)"
printf 'admitted_smoke=%s\nfive_step_job=%s\n' \
  "${FINOD_SMOKE_RUN_NAME}" "${five_job}"
