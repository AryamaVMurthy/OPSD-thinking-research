#!/usr/bin/env bash
# Validate complete diversity coverage, then launch eight-GPU confirmation.
#SBATCH --job-name=after-ch-diversity-cache
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
: "${CH_CONFIRM_CACHE_JOB_ID:?}" "${CH_CONFIRM_LIMIT:?}"
if [[ "${CH_CONFIRM_LIMIT}" != 1682 ]]; then
  echo "CH1 diversity confirmation requires exactly 1682 source identities" >&2
  exit 2
fi

cache_dir="${SCRATCH_ROOT}/ch_cache/ch-${CH_CONFIRM_LIMIT}-${CH_CONFIRM_CACHE_JOB_ID}"
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
from opsd_research.ch_confirmation import validate_confirmation_scale
from opsd_research.ch_dossier import accepted_teacher_dossiers
from opsd_research.training_data import load_math_cot_20k, partition_indices

manifest_path, summary_path = sys.argv[1:]
manifest = json.load(open(manifest_path, encoding="utf-8"))
summary = json.load(open(summary_path, encoding="utf-8"))
dossiers = accepted_teacher_dossiers(manifest_path)
if int(manifest["requested_examples"]) != 1682:
    raise SystemExit("confirmation manifest requested count is not 1682")
if int(summary["max_teacher_prompt_tokens"]) > 28672:
    raise SystemExit("confirmation teacher prompt exceeds training context")
rows = load_math_cot_20k(heldout_fraction=0.0)["train"]
train, _ = partition_indices(rows, 0.05)
event = validate_confirmation_scale(
    accepted_indices=dossiers,
    train_indices=train,
    requested_examples=1682,
    required_train_identities=1600,
)
event["max_teacher_prompt_tokens"] = summary["max_teacher_prompt_tokens"]
print(json.dumps(event, sort_keys=True))
PY

train="$(
  CH_DOSSIER_MANIFEST="${manifest}" \
  CH_CANDIDATE_CONFIG=reproductions/06_graf_opsd/configs/ch1-completion-aware-4096-diversity-confirmation.yaml \
  CH_CANDIDATE_ID=ch1-completion-aware-4096 \
  CH_MAX_STEPS=50 \
  CH_PORT=12984 \
  sbatch --parsable \
    --nodelist=node10 \
    --gres=gpu:8 \
    --cpus-per-task=64 \
    --mem=900G \
    --time=12:00:00 \
    --output="${SCRATCH_ROOT}/ch-opsd-train-%j.out" \
    reproductions/06_graf_opsd/run-ch-candidate.sbatch
)"
echo "ch1_diversity_training_job=${train}"
confirmation_run_name="ch-ch1-completion-aware-4096-50-${train}"
source_commit="$(git rev-parse HEAD)"
controller="$(
  sbatch --parsable \
    --dependency="afterok:${train}" \
    --output="${SCRATCH_ROOT}/slurm-after-ch-diversity-train-%j.out" \
    --export="ALL,CH_CONFIRM_TRAIN_JOB_ID=${train},CH_CONFIRM_RUN_NAME=${confirmation_run_name}" \
    reproductions/06_graf_opsd/after-ch-diversity-train.sbatch
)"
echo "ch1_diversity_training_controller=${controller}"
recovery="$(
  sbatch --parsable \
    --dependency="afternotok:${train}" \
    --output="${SCRATCH_ROOT}/slurm-recover-ch-diversity-train-%j.out" \
    --export="ALL,CH_CONFIRM_TRAIN_JOB_ID=${train},CH_CONFIRM_RUN_NAME=${confirmation_run_name},CH_DOSSIER_MANIFEST=${manifest},CH_RECOVERY_ATTEMPT=1,CH_EXPECTED_SOURCE_COMMIT=${source_commit}" \
    reproductions/06_graf_opsd/recover-ch-diversity-train.sbatch
)"
echo "ch1_diversity_training_recovery_controller=${recovery}"
