#!/usr/bin/env bash
set -euo pipefail

if (( $# != 2 )); then
  echo "usage: $0 OUTPUT_DIR MANIFEST_NAME" >&2
  exit 2
fi

output_dir="$1"
manifest_name="$2"
if [[ ! -d "${output_dir}" ]]; then
  echo "output directory does not exist: ${output_dir}" >&2
  exit 1
fi
if [[ "${manifest_name}" == */* ||
  ! "${manifest_name}" =~ ^artifact-manifest-[A-Za-z0-9._-]+\.sha256$ ]]; then
  echo "invalid artifact manifest name: ${manifest_name}" >&2
  exit 2
fi

mapfile -d '' files < <(
  cd "${output_dir}"
  find . -maxdepth 1 -type f \
    ! -name 'artifact-manifest-*.sha256' \
    ! -name '.artifact-manifest-*.tmp.*' \
    -printf '%P\0' |
    sort -z
)
if (( ${#files[@]} == 0 )); then
  echo "no artifacts found in ${output_dir}" >&2
  exit 1
fi

temporary="$(mktemp "${output_dir}/.${manifest_name}.tmp.XXXXXX")"
cleanup() {
  if [[ -n "${temporary:-}" && -f "${temporary}" ]]; then
    rm -f -- "${temporary}"
  fi
}
trap cleanup EXIT

(
  cd "${output_dir}"
  sha256sum "${files[@]}"
) > "${temporary}"
mv -f -- "${temporary}" "${output_dir}/${manifest_name}"
temporary=""
