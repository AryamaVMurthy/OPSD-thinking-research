#!/usr/bin/env bash

resolve_afterok_ids() {
  local job_id
  local record
  local state
  local exit_code
  local active_ids=()

  for job_id in "$@"; do
    if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
      printf 'invalid Slurm job ID: %s\n' "${job_id}" >&2
      return 2
    fi
    record="$(
      sacct -X -j "${job_id}" --format=JobIDRaw,State,ExitCode -n -P |
        awk -F'|' -v expected="${job_id}" '$1 == expected { print; exit }'
    )"
    IFS='|' read -r _ state exit_code <<< "${record}"
    state="${state%%+}"
    if [[ "${state}" == "COMPLETED" && "${exit_code}" == "0:0" ]]; then
      continue
    fi
    case "${state}" in
      PENDING | RUNNING | CONFIGURING | COMPLETING | SUSPENDED | RESIZING | REQUEUED)
        active_ids+=("${job_id}")
        continue
        ;;
    esac
    printf 'job %s is not a completed successful allocation\n' "${job_id}" >&2
    return 1
  done
  if (( ${#active_ids[@]} > 0 )); then
    local IFS=:
    printf '%s\n' "${active_ids[*]}"
  fi
}
