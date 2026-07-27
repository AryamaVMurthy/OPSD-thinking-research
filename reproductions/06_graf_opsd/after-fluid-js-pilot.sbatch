#!/usr/bin/env bash
# Verify the promoted Fluid-JS pilot and submit its paired AIME24 screen.
#SBATCH --job-name=after-fluid-js
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
: "${FLUID_PILOT_JOB_ID:?}"
: "${FLUID_EXPECTED_SOURCE_COMMIT:?}"

steps=12
run_name="graf-fluid-g4-js-128-${steps}-${FLUID_PILOT_JOB_ID}"
training_dir="${TRAIN_ROOT}/${run_name}"
summary="${training_dir}/training-summary.json"
log="${PROJECT_SOURCE}/logs/graf-routed-${FLUID_PILOT_JOB_ID}.out"
test -f "${summary}"
test -f "${log}"
test -f "${training_dir}/adapter_model.safetensors"
test "$(cat "${training_dir}/source-commit.txt")" = "${FLUID_EXPECTED_SOURCE_COMMIT}"
python3 -m opsd_research.slurm_accounting \
  --job-id "${FLUID_PILOT_JOB_ID}" \
  --output "${training_dir}/slurm-accounting.json"

python3 - "${summary}" "${log}" "${steps}" <<'PY'
import json
import math
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
log_path = sys.argv[2]
expected_steps = int(sys.argv[3])
integrity = summary["rollout_dump_integrity"]
if (
    not integrity["covers_latest_checkpoint"]
    or integrity["latest_checkpoint_step"] != expected_steps
    or summary["latest_logged_step"] != expected_steps
):
    raise SystemExit("Fluid-JS pilot lacks complete step-12 artifacts")
if len(summary["loss_history"]) != expected_steps:
    raise SystemExit("Fluid-JS pilot does not have exactly 12 optimizer records")
for row in summary["loss_history"]:
    if not all(
        math.isfinite(float(row[key])) for key in ("loss", "grad_norm")
    ):
        raise SystemExit("Fluid-JS pilot optimization is non-finite")
if max(
    item["max_memory_mib"] for item in summary["gpu_telemetry"]
) >= 44_000:
    raise SystemExit("Fluid-JS pilot exceeded the preregistered memory guard")

decoder = json.JSONDecoder()
events = []
for line in open(log_path, encoding="utf-8", errors="replace"):
    start = line.find('{"event":')
    if start < 0:
        continue
    try:
        serialized = (
            line[start:]
            .replace(":True", ":true")
            .replace(":False", ":false")
        )
        event, _ = decoder.raw_decode(serialized)
    except json.JSONDecodeError:
        continue
    events.append(event)
routing = [
    event for event in events
    if event.get("event") == "graf_viability_routing_enabled"
]
if not routing or not all(
    event.get("base_opsd_records") == 124
    and event.get("routed_examples") == 22
    and event.get("coverage_preserving") is True
    and event.get("student_answer_context") is False
    and event.get("teacher_empirical_continuation_context") is True
    for event in routing
):
    raise SystemExit("Fluid-JS coverage or teacher/student separation failed")
canonical = [
    event for event in events
    if event.get("event") == "canonical_divergence_loss"
]
if not canonical or any(
    event.get("objective") != "js"
    or not math.isfinite(float(event["value"]))
    or float(event["value"]) < -float(event["roundoff_tolerance"])
    for event in canonical
):
    raise SystemExit("Fluid-JS canonical objective failed")
branches = [
    event for event in events if event.get("event") == "graf_branch_loss"
]
if (
    not branches
    or not all(event.get("metric_scope") == "global_batch" for event in branches)
    or max(float(event["active_forks"]) for event in branches) <= 0.0
):
    raise SystemExit("Fluid-JS never activated a globally measured route")
adapter = [
    event for event in events if event.get("event") == "adapter_stability"
]
if len(adapter) != expected_steps or any(
    not math.isfinite(float(event["update_to_parameter_ratio"]))
    for event in adapter
):
    raise SystemExit("Fluid-JS adapter stability telemetry is incomplete")
PY

eval_run="fluid-g4-js-128-12-aime24-full-context-${FLUID_PILOT_JOB_ID}"
eval_job="$(
  CONFIG=reproductions/06_graf_opsd/configs/qwen3-4b-aime24-full-context-screen.yaml \
  EVAL_MODULE=math_eval \
  RUN_NAME="${eval_run}" \
  METHOD=fluid-g4-js \
  CHECKPOINT=step-12 \
  ADAPTER="${training_dir}" \
  sbatch --parsable infra/turing/run_eval.sbatch
)"
echo "fluid_js_aime24_eval_job=${eval_job}"
sbatch \
  --parsable \
  --dependency="afterok:${eval_job}" \
  --export="ALL,FLUID_PILOT_JOB_ID=${FLUID_PILOT_JOB_ID},FLUID_EVAL_RUN=${eval_run}" \
  reproductions/06_graf_opsd/after-fluid-js-eval.sbatch
