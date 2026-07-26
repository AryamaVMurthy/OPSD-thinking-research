#!/usr/bin/env bash
# Refresh a persistent live report for an active training job without using a
# GPU.  It is deliberately node-local so it can read the active node's scratch
# artifacts if those become available before the job copies final outputs.
#SBATCH --job-name=graf-live-monitor
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node03
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=18:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
module load u22/cuda/12.4
source "${HOME}/OPSD-thinking-research/infra/turing/common.sh"
cd "${PROJECT_SOURCE}"
: "${MONITOR_JOB_ID:?}" "${MONITOR_LOG:?}" "${MONITOR_MAX_COMPLETION_LENGTH:?}" "${MONITOR_REPORT_STEM:?}"

while squeue -h -j "${MONITOR_JOB_ID}" -o '%T' | grep -qx RUNNING; do
  python3 -m opsd_research.live_training_status \
    --training-log "${MONITOR_LOG}" \
    --max-completion-length "${MONITOR_MAX_COMPLETION_LENGTH}" \
    --output-json "${MONITOR_REPORT_STEM}.json" \
    --output-markdown "${MONITOR_REPORT_STEM}.md"
  sleep 60
done
python3 -m opsd_research.live_training_status \
  --training-log "${MONITOR_LOG}" \
  --max-completion-length "${MONITOR_MAX_COMPLETION_LENGTH}" \
  --output-json "${MONITOR_REPORT_STEM}.json" \
  --output-markdown "${MONITOR_REPORT_STEM}.md"
