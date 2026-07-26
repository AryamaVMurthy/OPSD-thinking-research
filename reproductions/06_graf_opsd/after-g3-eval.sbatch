#!/usr/bin/env bash
# Record G3's paired development result; locked sets remain untouched on failure.
#SBATCH --job-name=graf-after-g3-eval
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

: "${G3_PILOT_JOB_ID:?missing G3 pilot job ID}"
: "${G3_EVAL_RUN:?missing G3 evaluation run name}"
baseline="${RESULTS_ROOT}/graf-aime24-base-20260726"
treatment="${RESULTS_ROOT}/${G3_EVAL_RUN}"
test -f "${baseline}/summary.json"
test -f "${treatment}/summary.json"
python3 -m opsd_research.compare_math \
  --baseline-input "${baseline}"/generations.shard*.jsonl \
  --baseline-grades "${baseline}/official-grades.jsonl" \
  --treatment-input "${treatment}"/generations.shard*.jsonl \
  --treatment-grades "${treatment}/official-grades.jsonl" \
  --samples-per-problem 12 --output "${treatment}/paired-vs-base.json"
python3 - "${treatment}/paired-vs-base.json" "artifacts/graf_opsd/candidate-ledger.jsonl" <<'PY'
import json, sys
from pathlib import Path
from opsd_research.graf_contract import append_ledger
comparison = json.load(open(sys.argv[1], encoding="utf-8"))
delta = comparison["avg_at_12"]["delta"]
lower, upper = comparison["delta_bootstrap_95ci"]["avg_at_12"]
event = {"event": "g3_development_result", "candidate_id": "g3-viability-entropy-2048", "avg_at_12_delta": delta, "delta_ci_lower": lower, "delta_ci_upper": upper, "promotion_gate_passed": delta >= 0.03 and lower > 0 and upper > 0, "comparison_protocol": comparison["comparison_protocol"]}
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
  graph_manifest="${SCRATCH_ROOT}/graf_cache/g1-${CACHE_BUILD_JOB_ID}/manifest.json"
  viability_manifest="${SCRATCH_ROOT}/graf_cache/g2-viability-${G2_VIABILITY_ARRAY_JOB_ID}/manifest.json"
  full="$(sbatch --parsable --export="ALL,CANDIDATE_ID=g3-viability-entropy-2048,CANDIDATE_CONFIG=reproductions/06_graf_opsd/configs/g3-viability-entropy.yaml,METHOD=g3-viability-entropy,COMPLETION_LENGTH=2048,GRAF_GRAPH_CACHE_MANIFEST=${graph_manifest},GRAF_VIABILITY_MANIFEST=${viability_manifest}" reproductions/06_graf_opsd/promoted-full-train.sbatch)"
  echo "g3_full_confirmation_job=${full}"
  sbatch --parsable --dependency="afterok:${full}" --export="ALL,FULL_JOB_ID=${full},CANDIDATE_ID=g3-viability-entropy-2048,METHOD=g3-viability-entropy" reproductions/06_graf_opsd/after-promoted-full-train.sbatch
else
  # C1 is the independent 4,096-token OPSD control. It directly tests whether
  # the observed rollout truncation, rather than graph routing, explains the
  # lack of movement in G1--G3.
  python3 -m opsd_research.graf_autoresearch register \
    --policy reproductions/06_graf_opsd/configs/autoresearch.yaml \
    --config reproductions/06_graf_opsd/configs/c1-long-rollout-4096.yaml \
    --ledger artifacts/graf_opsd/candidate-ledger.jsonl \
    --candidate-id c1-long-rollout-4096 --parent-id c0-long-2048 --stage C1 \
    --hypothesis "Increase only the on-policy completion cap to reduce systematic early-thought truncation before adding further graph mechanisms." \
    --changed-field max_completion_length
  c1_smoke="$(sbatch --parsable reproductions/06_graf_opsd/c1-smoke.sbatch)"
  echo "c1_smoke_job=${c1_smoke}"
  sbatch --parsable --dependency="afterok:${c1_smoke}" \
    --export="ALL,C1_SMOKE_JOB_ID=${c1_smoke},CACHE_BUILD_JOB_ID=${CACHE_BUILD_JOB_ID},G2_VIABILITY_ARRAY_JOB_ID=${G2_VIABILITY_ARRAY_JOB_ID}" \
    reproductions/06_graf_opsd/after-c1-smoke.sbatch
fi
