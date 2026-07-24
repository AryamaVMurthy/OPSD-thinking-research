#!/usr/bin/env bash
set -euo pipefail

squeue -u "${USER}" -o "%.18i %.24j %.2t %.10M %.4D %R"
echo
sacct -u "${USER}" --starttime today \
  --format=JobID,JobName%28,State,Elapsed,ExitCode,AllocTRES%60
echo
for log in logs/*.out; do
  [[ -e "${log}" ]] || continue
  echo "===== ${log} ====="
  tail -n 12 "${log}"
done
