#!/usr/bin/env bash
# Compare the diversity confirmation and promote only a positive paired effect.
#SBATCH --job-name=after-ch-diversity-eval
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node10
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module load u22/cuda/12.4
source "${HOME}/OPSD-thinking-research/infra/turing/common.sh"
cd "${PROJECT_SOURCE}"
: "${CH_CONFIRM_EVAL_RUN:?}"

baseline="${RESULTS_ROOT}/graf-aime24-base-full-context-screen-20260727"
treatment="${RESULTS_ROOT}/${CH_CONFIRM_EVAL_RUN}"
python3 -m opsd_research.compare_math \
  --baseline-input "${baseline}"/generations.shard*.jsonl \
  --baseline-grades "${baseline}/official-grades.jsonl" \
  --treatment-input "${treatment}"/generations.shard*.jsonl \
  --treatment-grades "${treatment}/official-grades.jsonl" \
  --samples-per-problem 4 \
  --output "${treatment}/paired-vs-base.json"
python3 -m opsd_research.graf_development_report \
  --comparison "${treatment}/paired-vs-base.json" \
  --candidate-id ch1-completion-aware-4096-diversity-confirmation \
  --output "${treatment}/development-report.md"
python3 - "${treatment}/paired-vs-base.json" <<'PY'
import json, sys

comparison = json.load(open(sys.argv[1], encoding="utf-8"))
delta = float(comparison["avg_at_4"]["delta"])
print(json.dumps({
    "event": "ch1_diversity_confirmation_result",
    "avg_at_4_delta": delta,
    "decision": (
        "eligible-for-paper-scale-gate"
        if delta > 0
        else "stop-ch-development-no-positive-confirmation"
    ),
}, sort_keys=True))
PY
decision="$(python3 - "${treatment}/paired-vs-base.json" <<'PY'
import json, sys
comparison = json.load(open(sys.argv[1], encoding="utf-8"))
print("1" if float(comparison["avg_at_4"]["delta"]) > 0 else "0")
PY
)"
if [[ "${decision}" == 1 ]]; then
  promotion="$(
    sbatch --parsable \
      --nodelist=node10 \
      --output="${SCRATCH_ROOT}/slurm-promote-ch1-full-%j.out" \
      --export="ALL,CH_PILOT_EVAL_RUN=${CH_CONFIRM_EVAL_RUN}" \
      reproductions/06_graf_opsd/promote-ch1-full.sbatch
  )"
  echo "ch1_paper_scale_promotion_controller=${promotion}"
else
  echo "CH1 diversity confirmation did not clear the positive paired gate"
fi
