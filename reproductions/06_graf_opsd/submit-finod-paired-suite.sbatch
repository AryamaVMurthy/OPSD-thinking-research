#!/usr/bin/env bash
# Submit serialized untouched/candidate six-sample 32k evaluations and compares.
#SBATCH --job-name=submit-finod-eval
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
: "${FINOD_METHOD:?}"
: "${FINOD_CHECKPOINT:?}"
: "${FINOD_ADAPTER:?}"
: "${FINOD_ADAPTER_SHA256:?}"
: "${FINOD_EVAL_TAG:?}"
source "${HOME}/OPSD-thinking-research/infra/turing/common.sh"
cd "${PROJECT_SOURCE}"
test -z "$(git status --short)"

adapter_file="${FINOD_ADAPTER}/adapter_model.safetensors"
test -s "${adapter_file}"
actual_adapter_sha="$(sha256sum "${adapter_file}" | awk '{print $1}')"
if [[ "${actual_adapter_sha}" != "${FINOD_ADAPTER_SHA256}" ]]; then
  echo "FiNOD adapter checksum mismatch" >&2
  exit 1
fi

read -r -a benchmarks <<< "${FINOD_BENCHMARKS:-aime24 aime25 aime26}"
if (( ${#benchmarks[@]} == 0 )); then
  echo "FINOD_BENCHMARKS must name at least one benchmark" >&2
  exit 1
fi
declare -A seen=()
previous_job=""
for benchmark in "${benchmarks[@]}"; do
  case "${benchmark}" in
    aime24|aime25|aime26) ;;
    *)
      echo "Unsupported FiNOD benchmark ${benchmark}" >&2
      exit 1
      ;;
  esac
  if [[ -n "${seen[${benchmark}]:-}" ]]; then
    echo "Duplicate FiNOD benchmark ${benchmark}" >&2
    exit 1
  fi
  seen["${benchmark}"]=1

  config="reproductions/06_graf_opsd/configs/qwen3-4b-${benchmark}-six-32k.yaml"
  python3 -m opsd_research.cli validate-config "${config}"
  baseline_run="${FINOD_EVAL_TAG}-${benchmark}-base-six32k-${SLURM_JOB_ID}"
  candidate_run="${FINOD_EVAL_TAG}-${benchmark}-candidate-six32k-${SLURM_JOB_ID}"
  dependency=()
  if [[ -n "${previous_job}" ]]; then
    dependency=("--dependency=afterok:${previous_job}")
  fi
  baseline_job="$(
    sbatch --parsable \
      "${dependency[@]}" \
      --nodelist=node10 \
      --gres=gpu:8 \
      --ntasks=8 \
      --cpus-per-task=8 \
      --mem=900G \
      --export="ALL,PROJECT_SOURCE=${PROJECT_SOURCE},CONFIG=${config},EVAL_MODULE=math_eval,RUN_NAME=${baseline_run},METHOD=untouched,CHECKPOINT=1cfa9a7208912126459214e8b04321603b3df60c,ADAPTER=" \
      infra/turing/run_eval.sbatch
  )"
  candidate_job="$(
    sbatch --parsable \
      --dependency="afterok:${baseline_job}" \
      --nodelist=node10 \
      --gres=gpu:8 \
      --ntasks=8 \
      --cpus-per-task=8 \
      --mem=900G \
      --export="ALL,PROJECT_SOURCE=${PROJECT_SOURCE},CONFIG=${config},EVAL_MODULE=math_eval,RUN_NAME=${candidate_run},METHOD=${FINOD_METHOD},CHECKPOINT=${FINOD_CHECKPOINT},ADAPTER=${FINOD_ADAPTER}" \
      infra/turing/run_eval.sbatch
  )"
  compare_job="$(
    sbatch --parsable \
      --dependency="afterok:${candidate_job}" \
      --nodelist=node10 \
      --export="ALL,PROJECT_SOURCE=${PROJECT_SOURCE},BASELINE_RUN=${baseline_run},TREATMENT_RUN=${candidate_run},SAMPLES_PER_PROBLEM=6" \
      reproductions/06_graf_opsd/compare-finod-paired.sbatch
  )"
  printf '%s baseline_job=%s candidate_job=%s compare_job=%s baseline_run=%s candidate_run=%s\n' \
    "${benchmark}" "${baseline_job}" "${candidate_job}" "${compare_job}" \
    "${baseline_run}" "${candidate_run}"
  previous_job="${compare_job}"
done
