#!/usr/bin/env bash
set -euo pipefail

if (( $# != 2 )); then
  echo "usage: archive_eval.sh RUN_NAME JOB_ID" >&2
  exit 2
fi

run_name="$1"
job_id="$2"
if [[ ! "${run_name}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "invalid run name: ${run_name}" >&2
  exit 2
fi
if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
  echo "invalid Slurm job id: ${job_id}" >&2
  exit 2
fi

project_root="$(git rev-parse --show-toplevel)"
cd "${project_root}"

result_dir="artifacts/results/${run_name}"
log_path="artifacts/logs/opsd-eval-${job_id}.out"
telemetry_path="artifacts/telemetry/${run_name}-${job_id}.csv"
review_path="artifacts/reviews/${run_name}.md"
paired_review_path="artifacts/reviews/${run_name}-paired.md"
manual_review_path="artifacts/reviews/${run_name}-manual-notes.md"
manifest_name="artifact-manifest-local-official.sha256"

required_paths=(
  "${result_dir}"
  "${log_path}"
  "${telemetry_path}"
  "${review_path}"
  "${paired_review_path}"
  "${manual_review_path}"
  "${result_dir}/${manifest_name}"
)
for required_path in "${required_paths[@]}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "required accepted artifact is missing: ${required_path}" >&2
    exit 1
  fi
done

(
  cd "${result_dir}"
  sha256sum -c "${manifest_name}"
)

archive_dir="artifacts/archive"
archive_path="${archive_dir}/${run_name}.tar.gz"
mkdir -p "${archive_dir}"
first_archive="$(mktemp "${archive_dir}/.${run_name}.first.XXXXXX.tar.gz")"
second_archive="$(mktemp "${archive_dir}/.${run_name}.second.XXXXXX.tar.gz")"
cleanup() {
  rm -f -- "${first_archive:-}" "${second_archive:-}"
}
trap cleanup EXIT

archive_members=(
  "${result_dir}"
  "${log_path}"
  "${telemetry_path}"
  "${review_path}"
  "${paired_review_path}"
  "${manual_review_path}"
)
tar_options=(
  --sort=name
  --mtime=@0
  --owner=0
  --group=0
  --numeric-owner
)

tar "${tar_options[@]}" -czf "${first_archive}" "${archive_members[@]}"
tar "${tar_options[@]}" -czf "${second_archive}" "${archive_members[@]}"
first_hash="$(sha256sum "${first_archive}" | cut -d' ' -f1)"
second_hash="$(sha256sum "${second_archive}" | cut -d' ' -f1)"
if [[ "${first_hash}" != "${second_hash}" ]]; then
  echo "archive builds are not deterministic: ${first_hash} != ${second_hash}" >&2
  exit 1
fi
gzip -t "${first_archive}"

mv -f -- "${first_archive}" "${archive_path}"
first_archive=""
rm -f -- "${second_archive}"
second_archive=""
printf '%s  %s\n' "${first_hash}" "${archive_path}"
