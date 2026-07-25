#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: accept_math_eval.sh JOB_ID RUN_NAME BASELINE_RUN_NAME

Accept a completed Turing math evaluation into the local artifact store.
The command verifies Slurm completion, seals and checks a transfer manifest,
rescoring with pinned MathArena, validates paired samples, and generates a
rollout review packet. Manual review and final archiving remain separate gates.
EOF
  exit 2
}

if (( $# != 3 )); then
  usage
fi

job_id="$1"
run_name="$2"
baseline_run_name="$3"

if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
  echo "invalid Slurm job id: ${job_id}" >&2
  exit 2
fi
for value in "${run_name}" "${baseline_run_name}"; do
  if [[ ! "${value}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "invalid run name: ${value}" >&2
    exit 2
  fi
done

project_root="$(git rev-parse --show-toplevel)"
cd "${project_root}"

turing_target="${TURING_TARGET:-aryama.murthy@turing.iiit.ac.in}"
turing_host_key_alias="${TURING_HOST_KEY_ALIAS:-turing.iiit.ac.in}"
ssh_options=(
  -o BatchMode=yes
  -o ConnectTimeout=20
  -o ConnectionAttempts=2
  -o "HostKeyAlias=${turing_host_key_alias}"
)
rsync_shell="ssh -o BatchMode=yes -o ConnectTimeout=20 -o ConnectionAttempts=2 -o HostKeyAlias=${turing_host_key_alias}"

cluster_ssh() {
  ssh "${ssh_options[@]}" "${turing_target}" "$@"
}

retry_transfer() {
  local attempt
  for attempt in 1 2 3; do
    if "$@"; then
      return 0
    fi
    if (( attempt == 3 )); then
      echo "transfer failed after ${attempt} attempts: $*" >&2
      return 1
    fi
    echo "transfer attempt ${attempt} failed; retrying" >&2
    sleep 5
  done
}

accounting="$(
  cluster_ssh \
    "sacct -nX -j ${job_id} --format=State,ExitCode,NodeList -P | head -1"
)"
IFS='|' read -r state exit_code node_name <<< "${accounting}"
if [[ "${state}" != "COMPLETED" || "${exit_code}" != "0:0" ]]; then
  echo "refusing unaccepted Slurm state: ${accounting}" >&2
  exit 1
fi
if [[ ! "${node_name}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "invalid or missing execution node in accounting: ${accounting}" >&2
  exit 1
fi

remote_project="opsd-thinking-research"
remote_login_root="/scratch/${node_name}/${node_name}/aryama.murthy/${remote_project}"
remote_login_result="${remote_login_root}/results/${run_name}"
manifest_name="transfer-manifest-${job_id}.sha256"

local_result="artifacts/results/${run_name}"
local_baseline="artifacts/results/${baseline_run_name}"
local_log="artifacts/logs/opsd-eval-${job_id}.out"
local_telemetry="artifacts/telemetry/${run_name}-${job_id}.csv"
mkdir -p \
  "${local_result}" \
  artifacts/logs \
  artifacts/telemetry \
  artifacts/comparisons \
  artifacts/reviews

# Seal the login-visible scratch mirror rather than SSHing directly to the
# completed allocation node. Turing's pam_slurm_adopt policy correctly denies
# a direct node login once the job has exited, and the login mirror is
# read-only. Capture its file hashes over the authenticated connection, then
# persist that transfer manifest locally and validate the downloaded files
# against it below.
remote_manifest="$(
  cluster_ssh \
    "test -d '${remote_login_result}' && cd '${remote_login_result}' && find . -maxdepth 1 -type f ! -name 'transfer-manifest-*.sha256' -printf '%P\\0' | sort -z | xargs -0 sha256sum"
)"
if [[ -z "${remote_manifest}" ]]; then
  echo "refusing empty remote transfer manifest for ${remote_login_result}" >&2
  exit 1
fi
printf '%s\n' "${remote_manifest}" > "${local_result}/${manifest_name}"

retry_transfer rsync -a --partial --info=stats2 -e "${rsync_shell}" \
  "${turing_target}:${remote_login_result}/" "${local_result}/"
retry_transfer rsync -a --partial -e "${rsync_shell}" \
  "${turing_target}:OPSD-thinking-research/logs/opsd-eval-${job_id}.out" \
  "${local_log}"
retry_transfer rsync -a --partial -e "${rsync_shell}" \
  "${turing_target}:${remote_login_root}/telemetry/${run_name}-${job_id}.csv" \
  "${local_telemetry}"

(
  cd "${local_result}"
  sha256sum -c "${manifest_name}"
)

mapfile -t treatment_inputs < <(
  find "${local_result}" -maxdepth 1 -type f \
    -name 'generations.shard*.jsonl' | sort -V
)
mapfile -t baseline_inputs < <(
  find "${local_baseline}" -maxdepth 1 -type f \
    -name 'generations.shard*.jsonl' | sort -V
)
if (( ${#treatment_inputs[@]} == 0 || ${#baseline_inputs[@]} == 0 )); then
  echo "missing treatment or baseline generation shards" >&2
  exit 1
fi

python_bin="${project_root}/.venv/bin/python"
if [[ ! -x "${python_bin}" ]]; then
  echo "missing scorer Python: ${python_bin}" >&2
  exit 1
fi

"${python_bin}" -m opsd_research.rescore_matharena \
  --config "${local_result}/config.yaml" \
  --input "${treatment_inputs[@]}" \
  --grades-output "${local_result}/official-grades.jsonl" \
  --summary-output "${local_result}/summary.official.json"

"${python_bin}" -m opsd_research.compare_math \
  --baseline-input "${baseline_inputs[@]}" \
  --baseline-grades "${local_baseline}/official-grades.jsonl" \
  --treatment-input "${treatment_inputs[@]}" \
  --treatment-grades "${local_result}/official-grades.jsonl" \
  --samples-per-problem 12 \
  --bootstrap-samples 10000 \
  --output "artifacts/comparisons/${run_name}-vs-untouched.json"

"${python_bin}" -m opsd_research.review_rollouts \
  --input "${treatment_inputs[@]}" \
  --official-grades "${local_result}/official-grades.jsonl" \
  --output "artifacts/reviews/${run_name}.md" \
  --per-class 5 \
  --seed 42

"${python_bin}" -m opsd_research.review_math_changes \
  --baseline-input "${baseline_inputs[@]}" \
  --baseline-grades "${local_baseline}/official-grades.jsonl" \
  --treatment-input "${treatment_inputs[@]}" \
  --treatment-grades "${local_result}/official-grades.jsonl" \
  --output "artifacts/reviews/${run_name}-paired.md" \
  --per-class 3 \
  --seed 42

"${project_root}/infra/turing/artifact_manifest.sh" \
  "${local_result}" artifact-manifest-local-official.sha256

echo "accepted_job=${job_id}"
echo "accepted_run=${run_name}"
echo "result_dir=${local_result}"
echo "comparison=artifacts/comparisons/${run_name}-vs-untouched.json"
echo "review_packet=artifacts/reviews/${run_name}.md"
echo "paired_review_packet=artifacts/reviews/${run_name}-paired.md"
