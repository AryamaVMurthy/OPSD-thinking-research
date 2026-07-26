#!/usr/bin/env bash
# Resume an interrupted paper-scale CH1 run from its highest complete
# checkpoint. At most three automatic recovery attempts are allowed.
#SBATCH --job-name=recover-ch-full-train
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
: "${CH_FULL_TRAIN_JOB_ID:?}" "${CH_FULL_RUN_NAME:?}"
: "${CH_DOSSIER_MANIFEST:?}" "${CH_RECOVERY_ATTEMPT:?}"
: "${CH_EXPECTED_SOURCE_COMMIT:?}"
if ! [[ "${CH_RECOVERY_ATTEMPT}" =~ ^[1-3]$ ]]; then
  echo "CH_RECOVERY_ATTEMPT must be 1, 2, or 3" >&2
  exit 2
fi
if [[ "$(git rev-parse HEAD)" != "${CH_EXPECTED_SOURCE_COMMIT}" ]]; then
  echo "refusing to resume CH1 from a different source commit" >&2
  exit 2
fi

training_dir="${TRAIN_ROOT}/${CH_FULL_RUN_NAME}"
checkpoint="$(
  python3 - "${training_dir}" <<'PY'
import sys
from opsd_research.training_resume import latest_valid_checkpoint
checkpoint = latest_valid_checkpoint(sys.argv[1])
if checkpoint is None:
    raise SystemExit("no complete CH1 checkpoint exists for automatic recovery")
print(checkpoint)
PY
)"
echo "ch1_recovery_checkpoint=${checkpoint}"

resumed="$(
  CH_DOSSIER_MANIFEST="${CH_DOSSIER_MANIFEST}" \
  CH_CANDIDATE_CONFIG=reproductions/06_graf_opsd/configs/ch1-completion-aware-4096-full.yaml \
  CH_CANDIDATE_ID=ch1-completion-aware-4096-full \
  CH_MAX_STEPS=200 \
  CH_PORT=12986 \
  CH_TRAINING_LOG_ROOT="${SCRATCH_ROOT}" \
  CH_RUN_NAME="${CH_FULL_RUN_NAME}" \
  sbatch --parsable \
    --nodelist=node10 \
    --gres=gpu:8 \
    --cpus-per-task=64 \
    --mem=900G \
    --time=30:00:00 \
    --output="${SCRATCH_ROOT}/ch-opsd-train-%j.out" \
    reproductions/06_graf_opsd/run-ch-candidate.sbatch
)"
echo "ch1_resumed_training_job=${resumed}"
success="$(
  sbatch --parsable \
    --dependency="afterok:${resumed}" \
    --output="${SCRATCH_ROOT}/slurm-after-ch-full-train-%j.out" \
    --export="ALL,CH_FULL_TRAIN_JOB_ID=${CH_FULL_TRAIN_JOB_ID},CH_FULL_RUN_NAME=${CH_FULL_RUN_NAME}" \
    reproductions/06_graf_opsd/after-ch-full-train.sbatch
)"
echo "ch1_resumed_success_controller=${success}"

if (( CH_RECOVERY_ATTEMPT < 3 )); then
  next_attempt=$((CH_RECOVERY_ATTEMPT + 1))
  recovery="$(
    sbatch --parsable \
      --dependency="afternotok:${resumed}" \
      --output="${SCRATCH_ROOT}/slurm-recover-ch-full-train-%j.out" \
      --export="ALL,CH_FULL_TRAIN_JOB_ID=${CH_FULL_TRAIN_JOB_ID},CH_FULL_RUN_NAME=${CH_FULL_RUN_NAME},CH_DOSSIER_MANIFEST=${CH_DOSSIER_MANIFEST},CH_RECOVERY_ATTEMPT=${next_attempt},CH_EXPECTED_SOURCE_COMMIT=${CH_EXPECTED_SOURCE_COMMIT}" \
      reproductions/06_graf_opsd/recover-ch-full-train.sbatch
  )"
  echo "ch1_next_recovery_controller=${recovery}"
fi
