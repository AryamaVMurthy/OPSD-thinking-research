#!/usr/bin/env bash
set -euo pipefail

if (( $# < 1 || $# > 2 )); then
  echo "usage: $0 TRAIN_JOB_ID [ADAPTER_SMOKE_JOB_ID]" >&2
  exit 2
fi

train_job_id="$1"
smoke_job_id="${2:-}"
if [[ ! "${train_job_id}" =~ ^[0-9]+$ ]]; then
  echo "TRAIN_JOB_ID must be numeric" >&2
  exit 2
fi
if [[ -n "${smoke_job_id}" && ! "${smoke_job_id}" =~ ^[0-9]+$ ]]; then
  echo "ADAPTER_SMOKE_JOB_ID must be numeric" >&2
  exit 2
fi

project_source="${HOME}/OPSD-thinking-research"
cd "${project_source}"
source infra/turing/job_dependency.sh
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "refusing to submit post-training evaluations from a dirty tree" >&2
  exit 1
fi

adapter_root="/scratch/node10/${USER}/opsd-thinking-research/training/qwen3-1p7b-opsd-thinking"
afterok_jobs=("${train_job_id}")
if [[ -n "${smoke_job_id}" ]]; then
  afterok_jobs+=("${smoke_job_id}")
fi
afterok_ids="$(resolve_afterok_ids "${afterok_jobs[@]}")"
dependency_args=()
if [[ -n "${afterok_ids}" ]]; then
  dependency_args=(--dependency="afterok:${afterok_ids}")
fi
source_commit="$(git rev-parse HEAD)"
submission_id="$(date -u +%Y%m%dT%H%M%SZ)"
manifest="${project_source}/logs/posttrain-1p7b-${submission_id}.tsv"

printf 'source_commit\ttraining_job\tadapter_smoke_job\teval_job\tscore_job\trun_name\tcheckpoint\tconfig\tadapter\n' \
  > "${manifest}"

submit_eval() {
  local config="$1"
  local module="$2"
  local run_name="$3"
  local checkpoint="$4"
  local adapter="${adapter_root}/checkpoint-${checkpoint}"
  local job_id
  job_id="$(
    sbatch --parsable \
      "${dependency_args[@]}" \
      --export=ALL,CONFIG="${config}",EVAL_MODULE="${module}",RUN_NAME="${run_name}",ADAPTER="${adapter}",METHOD=opsd-standard-thinking,CHECKPOINT="${checkpoint}" \
      infra/turing/run_eval.sbatch
  )"
  printf '%s\n' "${job_id}"
}

for checkpoint in 50 100 150 200; do
  for benchmark in aime25 hmmt25; do
    config="reproductions/01_qwen3_thinking_math/configs/qwen3-1p7b-${benchmark}.yaml"
    run_name="opsd-qwen3-1p7b-step${checkpoint}-${benchmark}"
    eval_job="$(submit_eval "${config}" math_eval "${run_name}" "${checkpoint}")"
    printf '%s\t%s\t%s\t%s\t\t%s\t%s\t%s\t%s\n' \
      "${source_commit}" "${train_job_id}" "${smoke_job_id}" "${eval_job}" "${run_name}" \
      "${checkpoint}" "${config}" "${adapter_root}/checkpoint-${checkpoint}" \
      >> "${manifest}"
  done
done

checkpoint=200
config="reproductions/01_qwen3_thinking_math/configs/qwen3-1p7b-aime26.yaml"
run_name="opsd-qwen3-1p7b-step200-aime26"
eval_job="$(submit_eval "${config}" math_eval "${run_name}" "${checkpoint}")"
printf '%s\t%s\t%s\t%s\t\t%s\t%s\t%s\t%s\n' \
  "${source_commit}" "${train_job_id}" "${smoke_job_id}" "${eval_job}" "${run_name}" \
  "${checkpoint}" "${config}" "${adapter_root}/checkpoint-${checkpoint}" \
  >> "${manifest}"

config="reproductions/02_qwen3_thinking_livecodebench/configs/qwen3-1p7b-lcb-v6.yaml"
run_name="opsd-qwen3-1p7b-step200-lcb-v6"
lcb_job="$(submit_eval "${config}" lcb_eval "${run_name}" "${checkpoint}")"
score_job="$(
  sbatch --parsable \
    --dependency="afterok:${lcb_job}" \
    --export=ALL,CONFIG="${config}",RUN_NAME="${run_name}" \
    infra/turing/run_lcb_score.sbatch
)"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "${source_commit}" "${train_job_id}" "${smoke_job_id}" "${lcb_job}" "${score_job}" \
  "${run_name}" "${checkpoint}" "${config}" \
  "${adapter_root}/checkpoint-${checkpoint}" >> "${manifest}"

cat "${manifest}"
