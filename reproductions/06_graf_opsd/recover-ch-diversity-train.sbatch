#!/usr/bin/env bash
# Resume an interrupted 50-step diversity confirmation from its stable run
# directory. At most three automatic recovery attempts are allowed.
#SBATCH --job-name=recover-ch-diversity-train
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
: "${CH_CONFIRM_TRAIN_JOB_ID:?}" "${CH_CONFIRM_RUN_NAME:?}"
: "${CH_DOSSIER_MANIFEST:?}" "${CH_RECOVERY_ATTEMPT:?}"
: "${CH_EXPECTED_SOURCE_COMMIT:?}"
if ! [[ "${CH_RECOVERY_ATTEMPT}" =~ ^[1-3]$ ]]; then
  echo "CH_RECOVERY_ATTEMPT must be 1, 2, or 3" >&2
  exit 2
fi
if [[ "$(git rev-parse HEAD)" != "${CH_EXPECTED_SOURCE_COMMIT}" ]]; then
  echo "refusing to resume CH1 confirmation from a different source commit" >&2
  exit 2
fi

training_dir="${TRAIN_ROOT}/${CH_CONFIRM_RUN_NAME}"
checkpoint="$(
  python3 - "${training_dir}" <<'PY'
import sys
from opsd_research.training_resume import latest_valid_checkpoint

checkpoint = latest_valid_checkpoint(sys.argv[1])
print(checkpoint if checkpoint is not None else "restart-before-first-checkpoint")
PY
)"
echo "ch1_confirmation_recovery_checkpoint=${checkpoint}"

resumed="$(
  CH_DOSSIER_MANIFEST="${CH_DOSSIER_MANIFEST}" \
  CH_CANDIDATE_CONFIG=reproductions/06_graf_opsd/configs/ch1-completion-aware-4096-diversity-confirmation.yaml \
  CH_CANDIDATE_ID=ch1-completion-aware-4096 \
  CH_MAX_STEPS=50 \
  CH_PORT="$((12984 + CH_RECOVERY_ATTEMPT))" \
  CH_RUN_NAME="${CH_CONFIRM_RUN_NAME}" \
  sbatch --parsable \
    --nodelist=node10 \
    --gres=gpu:8 \
    --cpus-per-task=64 \
    --mem=900G \
    --time=12:00:00 \
    --output="${SCRATCH_ROOT}/ch-opsd-train-%j.out" \
    reproductions/06_graf_opsd/run-ch-candidate.sbatch
)"
echo "ch1_confirmation_resumed_training_job=${resumed}"
success="$(
  sbatch --parsable \
    --dependency="afterok:${resumed}" \
    --output="${SCRATCH_ROOT}/slurm-after-ch-diversity-train-%j.out" \
    --export="ALL,CH_CONFIRM_TRAIN_JOB_ID=${CH_CONFIRM_TRAIN_JOB_ID},CH_CONFIRM_RUN_NAME=${CH_CONFIRM_RUN_NAME}" \
    reproductions/06_graf_opsd/after-ch-diversity-train.sbatch
)"
echo "ch1_confirmation_resumed_success_controller=${success}"

if (( CH_RECOVERY_ATTEMPT < 3 )); then
  next_attempt=$((CH_RECOVERY_ATTEMPT + 1))
  recovery="$(
    sbatch --parsable \
      --dependency="afternotok:${resumed}" \
      --output="${SCRATCH_ROOT}/slurm-recover-ch-diversity-train-%j.out" \
      --export="ALL,CH_CONFIRM_TRAIN_JOB_ID=${CH_CONFIRM_TRAIN_JOB_ID},CH_CONFIRM_RUN_NAME=${CH_CONFIRM_RUN_NAME},CH_DOSSIER_MANIFEST=${CH_DOSSIER_MANIFEST},CH_RECOVERY_ATTEMPT=${next_attempt},CH_EXPECTED_SOURCE_COMMIT=${CH_EXPECTED_SOURCE_COMMIT}" \
      reproductions/06_graf_opsd/recover-ch-diversity-train.sbatch
  )"
  echo "ch1_confirmation_next_recovery_controller=${recovery}"
fi
