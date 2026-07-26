#!/usr/bin/env bash
# Record C1 fairly against the same paired untouched AIME24 generations.
#SBATCH --job-name=graf-after-c1-eval
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node03
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
event = {"event": "c1_development_result", "candidate_id": "c1-long-rollout-4096", "avg_at_12_delta": delta, "delta_ci_lower": lower, "delta_ci_upper": upper, "selection_metric": "official_aime24_avg_at_12", "comparison_protocol": comparison["comparison_protocol"]}
event["ledger_event_sha256"] = append_ledger(Path(sys.argv[2]), event)
print(json.dumps(event, sort_keys=True))
PY
# AIME24 Avg@12 itself is the requested development selection signal.  C1 is
# recorded for comparison, then the complete graph method runs regardless of
# its value—this avoids spending the graph budget on a statistical gate.
python3 -m opsd_research.graf_autoresearch register \
  --policy reproductions/06_graf_opsd/configs/autoresearch.yaml \
  --config reproductions/06_graf_opsd/configs/g4-viability-long-4096.yaml \
  --ledger artifacts/graf_opsd/candidate-ledger.jsonl \
  --candidate-id g4-graph-backtrack-entropy-4096 --parent-id c1-long-rollout-4096 --stage G4 \
  --hypothesis "Distil verifier-calibrated distributions over answer-masked teacher-proposed graph actions, including recovery actions and a target-adaptive entropy floor, at a 4,096-token on-policy rollout budget." \
  --changed-field full_graph_method
# Graph preparation consumes GPUs too. Build it only after the four-GPU C1
# pilot and its four-GPU official evaluation have released node03.
# 128 graphs gives the first complete dynamic method materially broader
# coverage than the historical 64-record pilot, while keeping forced-
# continuation scoring short enough for rapid autoresearch iteration.
cache_job="$(GRAPH_LIMIT=128 sbatch --parsable reproductions/06_graf_opsd/build-graph-cache.sbatch)"
echo "graph_cache_job=${cache_job}"
g2_array="$(sbatch --parsable --dependency="afterok:${cache_job}" --array=0-7%4 \
  --export="ALL,CACHE_BUILD_JOB_ID=${cache_job},G2_SHARD_COUNT=8" \
  reproductions/06_graf_opsd/build-g2-viability-shard.sbatch)"
echo "g2_viability_array_job=${g2_array}"
g2_merge="$(sbatch --parsable --dependency="afterok:${g2_array}" \
  --export="ALL,CACHE_BUILD_JOB_ID=${cache_job},G2_VIABILITY_ARRAY_JOB_ID=${g2_array},G2_SHARD_COUNT=8" \
  reproductions/06_graf_opsd/merge-g2-viability.sbatch)"
echo "g2_viability_merge_job=${g2_merge}"
graph_manifest="${SCRATCH_ROOT}/graf_cache/g1-${cache_job}/manifest.json"
viability_manifest="${SCRATCH_ROOT}/graf_cache/g2-viability-${g2_array}/manifest.json"
g4_smoke="$(sbatch --parsable --dependency="afterok:${g2_merge}" \
  --export="ALL,GRAF_GRAPH_CACHE_MANIFEST=${graph_manifest},GRAF_VIABILITY_MANIFEST=${viability_manifest},CACHE_BUILD_JOB_ID=${cache_job},G2_VIABILITY_ARRAY_JOB_ID=${g2_array}" \
  reproductions/06_graf_opsd/g4-smoke.sbatch)"
echo "g4_smoke_job=${g4_smoke}"
sbatch --parsable --dependency="afterok:${g4_smoke}" \
  --export="ALL,CACHE_BUILD_JOB_ID=${cache_job},G2_VIABILITY_ARRAY_JOB_ID=${g2_array},G4_SMOKE_JOB_ID=${g4_smoke}" \
  reproductions/06_graf_opsd/after-g4-smoke.sbatch
