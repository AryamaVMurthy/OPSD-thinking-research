#!/usr/bin/env bash
# Submit only after every build-g2-viability-shard task succeeded.
#SBATCH --job-name=graf-g2-viability-merge
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node03
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module load u22/cuda/12.4
source "${HOME}/OPSD-thinking-research/infra/turing/common.sh"
cd "${PROJECT_SOURCE}"

: "${CACHE_BUILD_JOB_ID:?missing immutable graph-cache build job ID}"
: "${G2_VIABILITY_ARRAY_JOB_ID:?missing completed viability array job ID}"
: "${G2_SHARD_COUNT:=8}"
: "${VIABILITY_CACHE_LABEL:=g2-viability}"
: "${GRAF_RECOVERY_CONDITIONED:=0}"
: "${VIABILITY_MAX_COMPLETION_TOKENS:=4096}"
graph_manifest="${SCRATCH_ROOT}/graf_cache/g1-${CACHE_BUILD_JOB_ID}/manifest.json"
target_root="${SCRATCH_ROOT}/graf_cache/${VIABILITY_CACHE_LABEL}-${G2_VIABILITY_ARRAY_JOB_ID}"

shards=()
trial_shards=()
for ((task=0; task<G2_SHARD_COUNT; task++)); do
  shards+=(--shard "${target_root}/part-${task}.jsonl")
  trial_shards+=(--trial-shard "${target_root}/part-${task}.trials.jsonl")
done
protocol="assistant_action_continuation_v1"
if [[ "${GRAF_RECOVERY_CONDITIONED}" == 1 ]]; then
  protocol="assistant_recovery_conditioned_continuation_v1"
elif [[ "${GRAF_RECOVERY_CONDITIONED}" != 0 ]]; then
  echo "GRAF_RECOVERY_CONDITIONED must be 0 or 1" >&2
  exit 2
fi
python3 -m opsd_research.merge_graf_viability \
  --graph-manifest "${graph_manifest}" \
  "${shards[@]}" \
  "${trial_shards[@]}" \
  --output "${target_root}/viability.jsonl" \
  --trials-output "${target_root}/viability.trials.jsonl" \
  --manifest "${target_root}/manifest.json" \
  --samples-per-action 2 --temperature 1.0 --max-completion-tokens "${VIABILITY_MAX_COMPLETION_TOKENS}" \
  --model Qwen/Qwen3-4B \
  --model-revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --seed 42 --forced-prefix-protocol "${protocol}"
