#!/usr/bin/env bash
# Record C1 fairly against the same paired untouched AIME24 generations.
#SBATCH --job-name=graf-after-c1-eval
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
: "${C1_PILOT_JOB_ID:?missing C1 pilot job}"
: "${C1_EVAL_RUN:?missing C1 evaluation run name}"
baseline="${RESULTS_ROOT}/graf-aime24-base-20260726"
treatment="${RESULTS_ROOT}/${C1_EVAL_RUN}"
test -f "${baseline}/summary.json"
test -f "${treatment}/summary.json"
python3 -m opsd_research.compare_math --baseline-input "${baseline}"/generations.shard*.jsonl --baseline-grades "${baseline}/official-grades.jsonl" --treatment-input "${treatment}"/generations.shard*.jsonl --treatment-grades "${treatment}/official-grades.jsonl" --samples-per-problem 12 --output "${treatment}/paired-vs-base.json"
python3 - "${treatment}/paired-vs-base.json" "artifacts/graf_opsd/candidate-ledger.jsonl" <<'PY'
import json, sys
from pathlib import Path
from opsd_research.graf_contract import append_ledger
comparison = json.load(open(sys.argv[1], encoding="utf-8"))
delta = comparison["avg_at_12"]["delta"]
lower, upper = comparison["delta_bootstrap_95ci"]["avg_at_12"]
event = {"event": "c1_development_result", "candidate_id": "c1-long-rollout-4096", "avg_at_12_delta": delta, "delta_ci_lower": lower, "delta_ci_upper": upper, "promotion_gate_passed": delta >= 0.03 and lower > 0 and upper > 0, "comparison_protocol": comparison["comparison_protocol"]}
event["ledger_event_sha256"] = append_ledger(Path(sys.argv[2]), event)
print(json.dumps(event, sort_keys=True))
PY
promote="$(python3 - "${treatment}/paired-vs-base.json" <<'PY'
import json, sys
comparison = json.load(open(sys.argv[1], encoding="utf-8"))
delta = comparison["avg_at_12"]["delta"]
lower, upper = comparison["delta_bootstrap_95ci"]["avg_at_12"]
print("1" if delta >= 0.03 and lower > 0 and upper > 0 else "0")
PY
)"
if [[ "${promote}" == "1" ]]; then
  adapter="${TRAIN_ROOT}/graf-c1-long-4096-50-${C1_PILOT_JOB_ID}"
  locked="$(sbatch --parsable --export="ALL,METHOD=opsd-long-4096,ADAPTER=${adapter}" reproductions/06_graf_opsd/submit-locked-evals.sbatch)"
  echo "c1_promoted_locked_submit_job=${locked}"
else
  echo "c1 did not clear the development gate; inspect the immutable ledger before proposing G4 recovery routing"
fi
