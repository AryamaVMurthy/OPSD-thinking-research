#!/usr/bin/env bash
set -euo pipefail

squeue -u "${USER}" -o "%.18i %.24j %.2t %.10M %.4D %R"
printf '\n'
sacct -u "${USER}" --starttime today \
  --format=JobID,JobName%28,State,Elapsed,ExitCode,AllocTRES%60
printf '\n'
df -h "${HOME}" /scratch/node10 2>/dev/null
printf '\n'
scratch_root="/scratch/node10/${USER}/opsd-thinking-research"
if [[ ! -d "${scratch_root}" ]]; then
  scratch_root="/scratch/node10/node10/${USER}/opsd-thinking-research"
fi
if [[ -d "${scratch_root}" ]]; then
  printf 'scratch_root=%s\n' "${scratch_root}"
  du -sh \
    "${scratch_root}/results" \
    "${scratch_root}/training" \
    "${scratch_root}/telemetry" \
    "${scratch_root}/archives" 2>/dev/null
  printf '\n'
  find "${scratch_root}/training" -maxdepth 2 -type d -name 'checkpoint-*' \
    -printf '%p\n' | sort -V
  printf '\n'
fi
for log in logs/*.out; do
  [[ -e "${log}" ]] || continue
  printf '===== %s =====\n' "${log}"
  tail -n 12 "${log}"
done
