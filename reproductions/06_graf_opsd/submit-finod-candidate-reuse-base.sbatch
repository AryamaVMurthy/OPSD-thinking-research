#!/usr/bin/env bash
# Evaluate one trained adapter against an existing paired-seed baseline.
#SBATCH --job-name=submit-finod-reuse
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
: "${FINOD_EVAL_TAG:?}"
: "${FINOD_BENCHMARK:?}"
: "${FINOD_BASELINE_RUN:?}"

source "${HOME}/OPSD-thinking-research/infra/turing/common.sh"
cd "${PROJECT_SOURCE}"
test -z "$(git status --short)"

case "${FINOD_BENCHMARK}" in
  aime24|aime25|aime26) ;;
  *)
    echo "Unsupported FiNOD benchmark ${FINOD_BENCHMARK}" >&2
    exit 1
    ;;
esac
adapter_file="${FINOD_ADAPTER}/adapter_model.safetensors"
test -s "${adapter_file}"
adapter_sha256="$(sha256sum "${adapter_file}" | awk '{print $1}')"
config="reproductions/06_graf_opsd/configs/qwen3-4b-${FINOD_BENCHMARK}-six-32k.yaml"
python3 -m opsd_research.cli validate-config "${config}"
test -s "${RESULTS_ROOT}/${FINOD_BASELINE_RUN}/official-grades.jsonl"

candidate_run="${FINOD_EVAL_TAG}-${FINOD_BENCHMARK}-candidate-six32k-${SLURM_JOB_ID}"
candidate_job="$(
  sbatch --parsable \
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
    --export="ALL,PROJECT_SOURCE=${PROJECT_SOURCE},BASELINE_RUN=${FINOD_BASELINE_RUN},TREATMENT_RUN=${candidate_run},SAMPLES_PER_PROBLEM=6" \
    reproductions/06_graf_opsd/compare-finod-paired.sbatch
)"
printf 'benchmark=%s adapter_sha256=%s candidate_job=%s compare_job=%s candidate_run=%s baseline_run=%s\n' \
  "${FINOD_BENCHMARK}" "${adapter_sha256}" "${candidate_job}" \
  "${compare_job}" "${candidate_run}" "${FINOD_BASELINE_RUN}"
