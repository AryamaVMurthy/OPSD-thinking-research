#!/usr/bin/env bash
# Validate the matched Fluid-G4 context-only control and launch its paired eval.
#SBATCH --job-name=after-fluid-context
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node03
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module load u22/cuda/12.4
source "${HOME}/OPSD-thinking-research/infra/turing/common.sh"
cd "${PROJECT_SOURCE}"
: "${FLUID_CONTEXT_JOB_ID:?}"

run_name="graf-fluid-g4-js-context-control-128-12-${FLUID_CONTEXT_JOB_ID}"
training_dir="${TRAIN_ROOT}/${run_name}"
summary="${training_dir}/training-summary.json"
training_log="${PROJECT_SOURCE}/logs/graf-routed-${FLUID_CONTEXT_JOB_ID}.out"
test -f "${summary}"
test -f "${training_dir}/checkpoint-12/adapter_model.safetensors"

python3 - "${summary}" "${training_log}" <<'PY'
import json
import math
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
if summary["latest_logged_step"] != 12 or len(summary["loss_history"]) != 12:
    raise SystemExit("context-only control lacks exactly 12 optimizer updates")
if not summary["loss_finite"] or any(
    not math.isfinite(float(row["loss"]))
    or not math.isfinite(float(row["grad_norm"]))
    for row in summary["loss_history"]
):
    raise SystemExit("context-only control optimization is non-finite")
if max(row["max_memory_mib"] for row in summary["gpu_telemetry"]) >= 44_000:
    raise SystemExit("context-only control exceeded the memory guard")

events = []
decoder = json.JSONDecoder()
for raw in open(sys.argv[2], encoding="utf-8", errors="replace"):
    for start, character in enumerate(raw):
        if character != "{":
            continue
        try:
            event, _ = decoder.raw_decode(raw[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)

controls = [
    event
    for event in events
    if event.get("event") == "fluid_teacher_context_control_enabled"
]
if len(controls) != 1:
    raise SystemExit("context-only control activation was not recorded exactly once")
control = controls[0]
if (
    control.get("base_opsd_records") != 124
    or not control.get("coverage_preserving")
    or not control.get("teacher_empirical_continuation_context")
    or control.get("action_routing_enabled")
):
    raise SystemExit("context-only control does not isolate action routing")
if any(event.get("event") == "graf_branch_loss" for event in events):
    raise SystemExit("context-only control unexpectedly optimized a route loss")
canonical = [
    event
    for event in events
    if event.get("event") == "canonical_divergence_loss"
]
if not canonical or any(
    event.get("objective") != "js"
    or not math.isfinite(float(event["value"]))
    or float(event["value"]) < -float(event["roundoff_tolerance"])
    for event in canonical
):
    raise SystemExit("context-only control canonical JS telemetry failed")
PY

eval_run="fluid-g4-js-context-control-128-12-aime24-full-context-${FLUID_CONTEXT_JOB_ID}"
eval_job="$(
  CONFIG=reproductions/06_graf_opsd/configs/qwen3-4b-aime24-full-context-screen.yaml \
  EVAL_MODULE=math_eval \
  RUN_NAME="${eval_run}" \
  METHOD=fluid-g4-js-context-control \
  CHECKPOINT=step-12 \
  ADAPTER="${training_dir}" \
  sbatch --parsable infra/turing/run_eval.sbatch
)"
echo "fluid_context_aime24_eval_job=${eval_job}"
sbatch \
  --parsable \
  --dependency="afterok:${eval_job}" \
  --export="ALL,FLUID_CONTEXT_JOB_ID=${FLUID_CONTEXT_JOB_ID},FLUID_CONTEXT_EVAL_RUN=${eval_run}" \
  reproductions/06_graf_opsd/after-fluid-context-control-eval.sbatch
