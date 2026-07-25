#!/usr/bin/env bash

vllm_port_for_task() {
  local job_id="${1:?Slurm job ID is required}"
  local task_id="${2:?Slurm task ID is required}"
  local base_port=20000
  local job_blocks=256
  local ports_per_task=16
  local tasks_per_job=8

  if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
    printf 'invalid Slurm job ID: %s\n' "${job_id}" >&2
    return 2
  fi
  if [[ ! "${task_id}" =~ ^[0-9]+$ ]] ||
    (( task_id < 0 || task_id >= tasks_per_job )); then
    printf 'invalid Slurm task ID: %s\n' "${task_id}" >&2
    return 2
  fi

  printf '%d\n' "$((base_port +
    (job_id % job_blocks) * ports_per_task * tasks_per_job +
    task_id * ports_per_task))"
}
