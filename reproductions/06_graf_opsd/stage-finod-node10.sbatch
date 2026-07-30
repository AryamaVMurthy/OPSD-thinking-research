#!/usr/bin/env bash
# Stage the exact FiNOD source and immutable G1 cache on node10 scratch.
#SBATCH --job-name=stage-finod
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node10
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail

root="/scratch/node10/${USER}/finod"
source_dir="${root}/source"
cache_dir="${root}/cache/g1-16326"
source_cache="${HOME}/finod-staging/g1-16326"
: "${FINOD_EXPECTED_COMMIT:?}"

if [[ -d "${source_dir}/.git" ]]; then
  test "$(git -C "${source_dir}" rev-parse HEAD)" = "${FINOD_EXPECTED_COMMIT}"
  test -z "$(git -C "${source_dir}" status --short)"
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
fi
test "$(git -C "${source_dir}" rev-parse HEAD)" = "${FINOD_EXPECTED_COMMIT}"
test -z "$(git -C "${source_dir}" status --short)"

mkdir -p "${cache_dir}"
cp "${source_cache}/graphs.jsonl" "${cache_dir}/graphs.jsonl"
cp "${source_cache}/manifest.json" "${cache_dir}/manifest.json"
sed -i \
  "s#\"cache\": \".*\"#\"cache\": \"${cache_dir}/graphs.jsonl\"#" \
  "${cache_dir}/manifest.json"
expected_sha="$(
  sed -n 's/.*"cache_sha256": "\([^"]*\)".*/\1/p' \
    "${cache_dir}/manifest.json"
)"
test -n "${expected_sha}"
test "$(sha256sum "${cache_dir}/graphs.jsonl" | awk '{print $1}')" = \
  "${expected_sha}"

printf 'source=%s\ncommit=%s\nmanifest=%s\n' \
  "${source_dir}" "${FINOD_EXPECTED_COMMIT}" "${cache_dir}/manifest.json"
