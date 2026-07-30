#!/usr/bin/env bash
# Stage one exact clean FiNOD source tree on the selected compute node.
#SBATCH --job-name=stage-finod-src
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail

: "${FINOD_EXPECTED_COMMIT:?}"
source "${HOME}/OPSD-thinking-research/infra/turing/common.sh"
root="${FINOD_SOURCE_ROOT:-${SCRATCH_ROOT}/finod}"
source_dir="${root}/source"

if [[ -d "${source_dir}/.git" ]]; then
  test -z "$(git -C "${source_dir}" status --short)"
  git -C "${source_dir}" fetch origin autoresearch/2026-07-31-finod
  git -C "${source_dir}" checkout --detach "${FINOD_EXPECTED_COMMIT}"
  git -C "${source_dir}" submodule update --init --recursive
elif [[ -e "${source_dir}" ]]; then
  echo "Refusing ambiguous existing FiNOD source: ${source_dir}" >&2
  exit 1
else
  mkdir -p "${root}"
  git clone \
    --recurse-submodules \
    --branch autoresearch/2026-07-31-finod \
    https://github.com/AryamaVMurthy/OPSD-thinking-research.git \
    "${source_dir}"
  git -C "${source_dir}" checkout --detach "${FINOD_EXPECTED_COMMIT}"
fi
test "$(git -C "${source_dir}" rev-parse HEAD)" = "${FINOD_EXPECTED_COMMIT}"
test -z "$(git -C "${source_dir}" status --short)"
printf 'source=%s\ncommit=%s\n' "${source_dir}" "${FINOD_EXPECTED_COMMIT}"
