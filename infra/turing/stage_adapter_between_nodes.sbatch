#!/usr/bin/env bash
# Stage a root-only, checksum-pinned adapter bundle between node-local disks.
# Submit with both nodes allocated and the source node listed first.
#SBATCH --job-name=stage-adapter
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

: "${SOURCE_NODE:?}"
: "${DESTINATION_NODE:?}"
: "${SOURCE_ADAPTER:?}"
: "${DESTINATION_ADAPTER:?}"
: "${EXPECTED_ADAPTER_SHA256:?}"

if [[ "$(hostname -s)" != "${SOURCE_NODE}" ]]; then
  echo "batch host must be source node ${SOURCE_NODE}, found $(hostname -s)" >&2
  exit 1
fi

PROJECT_SOURCE="${HOME}/OPSD-thinking-research"
source "${PROJECT_SOURCE}/infra/turing/common.sh"

source_archive="/tmp/opsd-adapter-source-${SLURM_JOB_ID}.tar"
received_archive="/tmp/opsd-adapter-received-${SLURM_JOB_ID}.tar"

cleanup() {
  rm -f -- "${source_archive}"
  srun --nodes=2 --ntasks=2 \
    rm -f -- "${received_archive}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

python3 -m opsd_research.adapter_bundle pack \
  --source "${SOURCE_ADAPTER}" \
  --archive "${source_archive}" \
  --expected-sha256 "${EXPECTED_ADAPTER_SHA256}"

sbcast --force "${source_archive}" "${received_archive}"

srun --nodes=1 --ntasks=1 --nodelist="${DESTINATION_NODE}" \
  bash -lc '
    set -euo pipefail
    source "${HOME}/OPSD-thinking-research/infra/turing/common.sh"
    python3 -m opsd_research.adapter_bundle unpack \
      --archive "$1" \
      --destination "$2" \
      --expected-sha256 "$3"
  ' bash \
  "${received_archive}" \
  "${DESTINATION_ADAPTER}" \
  "${EXPECTED_ADAPTER_SHA256}"

srun --nodes=1 --ntasks=1 --nodelist="${DESTINATION_NODE}" \
  sha256sum "${DESTINATION_ADAPTER}/adapter_model.safetensors"
