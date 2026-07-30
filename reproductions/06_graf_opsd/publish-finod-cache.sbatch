#!/usr/bin/env bash
# Persist a completed node-local representative cache in small shared storage.
#SBATCH --job-name=publish-finod
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node03
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail

: "${FINOD_CACHE_JOB_ID:?}"
source "${HOME}/OPSD-thinking-research/infra/turing/common.sh"
source_dir="${SCRATCH_ROOT}/finod_cache/representative-${FINOD_CACHE_JOB_ID}"
destination="${HOME}/OPSD-thinking-research/artifacts/finod-cache-${FINOD_CACHE_JOB_ID}"

test -s "${source_dir}/graphs.jsonl"
test -s "${source_dir}/manifest.json"
test -s "${source_dir}/source-commit.txt"
if [[ -e "${destination}" ]]; then
  echo "Refusing to overwrite published cache ${destination}" >&2
  exit 1
fi
mkdir -p "${destination}"
cp "${source_dir}/graphs.jsonl" "${destination}/graphs.jsonl"
cp "${source_dir}/manifest.json" "${destination}/manifest.json"
cp "${source_dir}/source-commit.txt" "${destination}/source-commit.txt"
sed -i \
  "s#\"cache\": \".*\"#\"cache\": \"${destination}/graphs.jsonl\"#" \
  "${destination}/manifest.json"
expected_sha="$(
  sed -n 's/.*"cache_sha256": "\([^"]*\)".*/\1/p' \
    "${destination}/manifest.json"
)"
test -n "${expected_sha}"
test "$(sha256sum "${destination}/graphs.jsonl" | awk '{print $1}')" = \
  "${expected_sha}"
(
  cd "${destination}"
  sha256sum graphs.jsonl manifest.json source-commit.txt \
    > artifact-manifest.sha256
)
accepted="$(
  sed -n 's/.*"accepted_examples": \([0-9][0-9]*\).*/\1/p' \
    "${destination}/manifest.json"
)"
printf 'published_cache=%s\naccepted_examples=%s\n' \
  "${destination}/manifest.json" "${accepted}"
