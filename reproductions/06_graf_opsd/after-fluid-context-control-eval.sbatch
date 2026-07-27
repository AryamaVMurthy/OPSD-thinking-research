#!/usr/bin/env bash
# Compare context-only Fluid-G4 to both the immutable base and full Fluid-G4.
#SBATCH --job-name=after-fluid-context-eval
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
: "${FLUID_CONTEXT_EVAL_RUN:?}"

baseline="${RESULTS_ROOT}/graf-aime24-base-full-context-screen-20260727"
full_fluid="${RESULTS_ROOT}/fluid-g4-js-128-12-aime24-full-context-16595"
context="${RESULTS_ROOT}/${FLUID_CONTEXT_EVAL_RUN}"
for directory in "${baseline}" "${full_fluid}" "${context}"; do
  test -f "${directory}/summary.json"
  test -f "${directory}/official-grades.jsonl"
done

compare() {
  local reference="$1"
  local output="$2"
  python3 -m opsd_research.compare_math \
    --baseline-input "${reference}"/generations.shard*.jsonl \
    --baseline-grades "${reference}/official-grades.jsonl" \
    --treatment-input "${context}"/generations.shard*.jsonl \
    --treatment-grades "${context}/official-grades.jsonl" \
    --samples-per-problem 4 \
    --output "${output}"
}
compare "${baseline}" "${context}/paired-vs-base.json"
compare "${full_fluid}" "${context}/paired-vs-full-fluid.json"

python3 - "${context}/paired-vs-base.json" "${context}/paired-vs-full-fluid.json" <<'PY'
import json
import sys

vs_base = json.load(open(sys.argv[1], encoding="utf-8"))
vs_full = json.load(open(sys.argv[2], encoding="utf-8"))
base_delta = float(vs_base["avg_at_4"]["delta"])
route_delta = float(vs_full["avg_at_4"]["delta"])
if base_delta > 0:
    decision = "remove-route-and-replicate-context-only"
elif route_delta > 0:
    decision = "route-harms-but-context-still-needs-outcome-protection"
else:
    decision = "full-response-context-needs-outcome-protection"
print(json.dumps({
    "event": "fluid_context_attribution_result",
    "avg_at_4_delta_vs_base": base_delta,
    "avg_at_4_delta_vs_full_fluid": route_delta,
    "decision": decision,
    "locked_benchmarks_used": False,
}, sort_keys=True))
PY
