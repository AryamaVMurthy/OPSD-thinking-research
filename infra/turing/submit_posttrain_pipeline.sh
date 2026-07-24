#!/usr/bin/env bash
set -euo pipefail

if (( $# != 2 )); then
  echo "usage: $0 TRAIN_1P7B_JOB_ID TRAIN_4B_JOB_ID" >&2
  exit 2
fi

train_1p7b="$1"
train_4b="$2"
for job_id in "${train_1p7b}" "${train_4b}"; do
  if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
    echo "training job IDs must be numeric" >&2
    exit 2
  fi
done

project_source="${HOME}/OPSD-thinking-research"
cd "${project_source}"
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "refusing to submit post-training pipeline from a dirty tree" >&2
  exit 1
fi

source_commit="$(git rev-parse HEAD)"
submission_id="$(date -u +%Y%m%dT%H%M%SZ)"
pipeline_manifest="${project_source}/logs/posttrain-pipeline-${submission_id}.tsv"
scratch_training="/scratch/node10/${USER}/opsd-thinking-research/training"

submit_smoke() {
  local training_job="$1"
  local config="$2"
  local run_name="$3"
  local adapter="$4"
  sbatch --parsable \
    --dependency="afterok:${training_job}" \
    --export=ALL,CONFIG="${config}",RUN_NAME="${run_name}",ADAPTER="${adapter}",CHECKPOINT=200 \
    infra/turing/smoke_adapter_inference.sbatch
}

smoke_1p7b_name="adapter-smoke-qwen3-1p7b-step200-${submission_id}"
smoke_1p7b="$(
  submit_smoke \
    "${train_1p7b}" \
    reproductions/01_qwen3_thinking_math/configs/qwen3-1p7b-aime25.yaml \
    "${smoke_1p7b_name}" \
    "${scratch_training}/qwen3-1p7b-opsd-thinking/checkpoint-200"
)"

smoke_4b_name="adapter-smoke-qwen3-4b-step200-${submission_id}"
smoke_4b="$(
  submit_smoke \
    "${train_4b}" \
    reproductions/01_qwen3_thinking_math/configs/qwen3-4b-aime25.yaml \
    "${smoke_4b_name}" \
    "${scratch_training}/qwen3-4b-opsd-thinking/checkpoint-200"
)"

{
  printf 'source_commit\tmodel\ttraining_job\tadapter_smoke_job\tadapter_smoke_run\n'
  printf '%s\tqwen3-1p7b\t%s\t%s\t%s\n' \
    "${source_commit}" "${train_1p7b}" "${smoke_1p7b}" "${smoke_1p7b_name}"
  printf '%s\tqwen3-4b\t%s\t%s\t%s\n' \
    "${source_commit}" "${train_4b}" "${smoke_4b}" "${smoke_4b_name}"
} > "${pipeline_manifest}"

printf 'pipeline_manifest=%s\n' "${pipeline_manifest}"
cat "${pipeline_manifest}"
infra/turing/submit_posttrain_1p7b.sh "${train_1p7b}" "${smoke_1p7b}"
infra/turing/submit_posttrain_4b.sh "${train_4b}" "${smoke_4b}"
