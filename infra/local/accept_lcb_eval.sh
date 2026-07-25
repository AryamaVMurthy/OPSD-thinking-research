#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: accept_lcb_eval.sh GENERATION_JOB_ID SCORE_JOB_ID RUN_NAME BASELINE_RUN_NAME

Accept a generated and officially scored Turing LiveCodeBench evaluation.
Both Slurm jobs must have completed successfully. The command seals and checks
the remote artifacts, retrieves logs and telemetry, verifies paired samples,
computes paired pass@k deltas, and creates a rollout review packet. Manual
review and final archiving remain separate gates.
EOF
  exit 2
}

if (( $# != 4 )); then
  usage
fi

generation_job_id="$1"
score_job_id="$2"
run_name="$3"
baseline_run_name="$4"

for job_id in "${generation_job_id}" "${score_job_id}"; do
  if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
    echo "invalid Slurm job id: ${job_id}" >&2
    exit 2
  fi
done
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

accepted_node=""
for job_id in "${generation_job_id}" "${score_job_id}"; do
  accounting="$(
    cluster_ssh \
      "sacct -nX -j ${job_id} --format=State,ExitCode,NodeList -P | head -1"
  )"
  IFS='|' read -r state exit_code node_name <<< "${accounting}"
  if [[ "${state}" != "COMPLETED" || "${exit_code}" != "0:0" ]]; then
    echo "refusing unaccepted Slurm state for ${job_id}: ${accounting}" >&2
    exit 1
  fi
  if [[ ! "${node_name}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "invalid execution node for ${job_id}: ${accounting}" >&2
    exit 1
  fi
  if [[ -n "${accepted_node}" && "${accepted_node}" != "${node_name}" ]]; then
    echo "generation and scoring ran on different scratch nodes" >&2
    exit 1
  fi
  accepted_node="${node_name}"
done

remote_project="opsd-thinking-research"
remote_compute_root="/scratch/${accepted_node}/aryama.murthy/${remote_project}"
remote_login_root="/scratch/${accepted_node}/${accepted_node}/aryama.murthy/${remote_project}"
remote_result="${remote_compute_root}/results/${run_name}"
remote_login_result="${remote_login_root}/results/${run_name}"
manifest_name="transfer-manifest-${score_job_id}.sha256"

cluster_ssh \
  "ssh ${accepted_node} \"test -s '${remote_result}/graded.jsonl' && test -s '${remote_result}/summary.json' && cd '${remote_result}' && find . -maxdepth 1 -type f ! -name 'transfer-manifest-*.sha256' -printf '%P\\\\0' | sort -z | xargs -0 sha256sum > '${manifest_name}' && sha256sum -c '${manifest_name}'\""

local_result="artifacts/results/${run_name}"
local_generation_log="artifacts/logs/opsd-eval-${generation_job_id}.out"
local_score_log="artifacts/logs/opsd-lcb-score-${score_job_id}.out"
local_telemetry="artifacts/telemetry/${run_name}-${generation_job_id}.csv"
mkdir -p \
  "${local_result}" \
  artifacts/logs \
  artifacts/telemetry \
  artifacts/comparisons \
  artifacts/reviews

retry_transfer rsync -a --partial --info=stats2 -e "${rsync_shell}" \
  "${turing_target}:${remote_login_result}/" "${local_result}/"
retry_transfer rsync -a --partial -e "${rsync_shell}" \
  "${turing_target}:OPSD-thinking-research/logs/opsd-eval-${generation_job_id}.out" \
  "${local_generation_log}"
retry_transfer rsync -a --partial -e "${rsync_shell}" \
  "${turing_target}:OPSD-thinking-research/logs/opsd-lcb-score-${score_job_id}.out" \
  "${local_score_log}"
retry_transfer rsync -a --partial -e "${rsync_shell}" \
  "${turing_target}:${remote_login_root}/telemetry/${run_name}-${generation_job_id}.csv" \
  "${local_telemetry}"

(
  cd "${local_result}"
  sha256sum -c "${manifest_name}"
)

baseline_archive="artifacts/archive/${baseline_run_name}.tar.gz"
if [[ ! -s "${baseline_archive}" ]]; then
  echo "missing untouched baseline archive: ${baseline_archive}" >&2
  exit 1
fi
baseline_temp="$(mktemp -d)"
cleanup() {
  rm -rf -- "${baseline_temp}"
}
trap cleanup EXIT
tar -xzf "${baseline_archive}" -C "${baseline_temp}" \
  "${baseline_run_name}/graded.jsonl"
baseline_graded="${baseline_temp}/${baseline_run_name}/graded.jsonl"

python_bin="${project_root}/.venv/bin/python"
if [[ ! -x "${python_bin}" ]]; then
  echo "missing scorer Python: ${python_bin}" >&2
  exit 1
fi

"${python_bin}" -m opsd_research.compare_lcb \
  --baseline-graded "${baseline_graded}" \
  --treatment-graded "${local_result}/graded.jsonl" \
  --samples-per-problem 10 \
  --bootstrap-samples 10000 \
  --output "artifacts/comparisons/${run_name}-vs-untouched.json"

"${python_bin}" -m opsd_research.review_rollouts \
  --input "${local_result}/graded.jsonl" \
  --output "artifacts/reviews/${run_name}.md" \
  --per-class 5 \
  --seed 42

"${project_root}/infra/turing/artifact_manifest.sh" \
  "${local_result}" artifact-manifest-local-official.sha256

echo "accepted_generation_job=${generation_job_id}"
echo "accepted_score_job=${score_job_id}"
echo "accepted_run=${run_name}"
echo "result_dir=${local_result}"
echo "comparison=artifacts/comparisons/${run_name}-vs-untouched.json"
echo "review_packet=artifacts/reviews/${run_name}.md"
