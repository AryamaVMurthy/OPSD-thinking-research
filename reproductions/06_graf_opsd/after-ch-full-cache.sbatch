#!/usr/bin/env bash
# Gate a paper-scale CH cache and launch fresh eight-GPU training.
#SBATCH --job-name=after-ch-full-cache
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node10
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module load u22/cuda/12.4
source "${HOME}/OPSD-thinking-research/infra/turing/common.sh"
cd "${PROJECT_SOURCE}"
: "${CH_FULL_CACHE_JOB_ID:?}" "${CH_FULL_LIMIT:?}"
if [[ "${CH_FULL_LIMIT}" != 6912 ]]; then
  echo "paper-scale CH1 cache must contain exactly 6912 requested identities" >&2
  exit 2
fi

cache_dir="${SCRATCH_ROOT}/ch_cache/ch-${CH_FULL_LIMIT}-${CH_FULL_CACHE_JOB_ID}"
manifest="${cache_dir}/manifest.json"
summary="${cache_dir}/cache-summary.json"
test -f "${manifest}"
test -f "${summary}"
(
  cd "${cache_dir}"
  sha256sum -c artifact-manifest.sha256
)
python3 - "${manifest}" "${summary}" <<'PY'
import json, sys
from opsd_research.ch_dossier import accepted_teacher_dossiers
from opsd_research.training_data import load_math_cot_20k, partition_indices

manifest_path, summary_path = sys.argv[1:]
manifest = json.load(open(manifest_path, encoding="utf-8"))
summary = json.load(open(summary_path, encoding="utf-8"))
dossiers = accepted_teacher_dossiers(manifest_path)
if int(manifest["requested_examples"]) != 6912 or len(dossiers) != 6912:
    raise SystemExit("full CH1 cache must retain all 6912 identities")
if int(summary["max_teacher_prompt_tokens"]) > 28672:
    raise SystemExit("full CH1 teacher prompt exceeds the configured context")
rows = load_math_cot_20k(heldout_fraction=0.0)["train"]
train, _ = partition_indices(rows, 0.05)
eligible = len(set(train).intersection(dossiers))
if eligible < 6400:
    raise SystemExit(
        f"only {eligible} train identities remain; 6400 are required "
        "for one non-recycled 200-step pass"
    )
print(json.dumps({
    "event": "ch1_full_cache_gate",
    "requested": 6912,
    "accepted": len(dossiers),
    "eligible_train_identities": eligible,
    "required_first_pass_identities": 6400,
    "max_teacher_prompt_tokens": summary["max_teacher_prompt_tokens"],
}, sort_keys=True))
PY

train="$(
  CH_DOSSIER_MANIFEST="${manifest}" \
  CH_CANDIDATE_CONFIG=reproductions/06_graf_opsd/configs/ch1-completion-aware-4096-full.yaml \
  CH_CANDIDATE_ID=ch1-completion-aware-4096-full \
  CH_MAX_STEPS=200 \
  CH_PORT=12986 \
  CH_TRAINING_LOG_ROOT="${SCRATCH_ROOT}" \
  sbatch --parsable \
    --nodelist=node10 \
    --gres=gpu:8 \
    --cpus-per-task=64 \
    --mem=900G \
    --time=30:00:00 \
    --output="${SCRATCH_ROOT}/ch-opsd-train-%j.out" \
    reproductions/06_graf_opsd/run-ch-candidate.sbatch
)"
echo "ch1_full_training_job=${train}"
full_run_name="ch-ch1-completion-aware-4096-full-200-${train}"
source_commit="$(git rev-parse HEAD)"
controller="$(
  sbatch --parsable \
    --dependency="afterok:${train}" \
    --output="${SCRATCH_ROOT}/slurm-after-ch-full-train-%j.out" \
    --export="ALL,CH_FULL_TRAIN_JOB_ID=${train},CH_FULL_RUN_NAME=${full_run_name}" \
    reproductions/06_graf_opsd/after-ch-full-train.sbatch
)"
echo "ch1_full_training_controller=${controller}"
recovery="$(
  sbatch --parsable \
    --dependency="afternotok:${train}" \
    --output="${SCRATCH_ROOT}/slurm-recover-ch-full-train-%j.out" \
    --export="ALL,CH_FULL_TRAIN_JOB_ID=${train},CH_FULL_RUN_NAME=${full_run_name},CH_DOSSIER_MANIFEST=${manifest},CH_RECOVERY_ATTEMPT=1,CH_EXPECTED_SOURCE_COMMIT=${source_commit}" \
    reproductions/06_graf_opsd/recover-ch-full-train.sbatch
)"
echo "ch1_full_training_recovery_controller=${recovery}"
