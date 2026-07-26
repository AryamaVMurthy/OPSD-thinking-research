#!/usr/bin/env bash
# Produce the official 12-sample paper-scale CH1-vs-untouched AIME24 result.
#SBATCH --job-name=after-ch-full-aime24
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
: "${CH_FULL_TRAIN_JOB_ID:?}" "${CH_FULL_TREATMENT_RUN:?}" "${CH_FULL_BASELINE_RUN:?}"
treatment="${RESULTS_ROOT}/${CH_FULL_TREATMENT_RUN}"
baseline="${RESULTS_ROOT}/${CH_FULL_BASELINE_RUN}"
test -f "${treatment}/summary.json"
test -f "${baseline}/summary.json"
python3 -m opsd_research.compare_math \
  --baseline-input "${baseline}"/generations.shard*.jsonl \
  --baseline-grades "${baseline}/official-grades.jsonl" \
  --treatment-input "${treatment}"/generations.shard*.jsonl \
  --treatment-grades "${treatment}/official-grades.jsonl" \
  --samples-per-problem 12 \
  --output "${treatment}/paired-vs-base.json"
python3 -m opsd_research.graf_development_report \
  --comparison "${treatment}/paired-vs-base.json" \
  --candidate-id ch1-completion-aware-4096-full \
  --output "${treatment}/official-aime24-report.md"
python3 - "${treatment}/paired-vs-base.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1], encoding="utf-8"))
d = float(c["avg_at_12"]["delta"])
lo, hi = c["delta_bootstrap_95ci"]["avg_at_12"]
print(json.dumps({
    "event": "ch1_full_official_aime24_result",
    "avg_at_12_delta": d,
    "delta_ci_lower": lo,
    "delta_ci_upper": hi,
    "locked_evaluation_gate": (
        "eligible" if d > 0 else "stop-no-positive-full-scale-effect"
    ),
}, sort_keys=True))
PY

locked_gate="$(
  python3 - "${treatment}/paired-vs-base.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1], encoding="utf-8"))
print("yes" if float(c["avg_at_12"]["delta"]) > 0 else "no")
PY
)"
if [[ "${locked_gate}" != yes ]]; then
  echo "CH1 full AIME24 effect is not positive; locked benchmarks remain untouched"
  exit 0
fi

method="ch1-completion-aware-4096-full"
training_dir="${TRAIN_ROOT}/ch-${method}-200-${CH_FULL_TRAIN_JOB_ID}"
test -f "${training_dir}/adapter_model.safetensors"
previous=""
locked_runs=()
for benchmark in aime25 aime26; do
  config="reproductions/01_qwen3_thinking_math/configs/qwen3-4b-${benchmark}.yaml"
  treatment_run="${method}-200-${benchmark}-official-${CH_FULL_TRAIN_JOB_ID}"
  baseline_run="qwen3-4b-base-${benchmark}-official-${CH_FULL_TRAIN_JOB_ID}"
  dependency=()
  if [[ -n "${previous}" ]]; then
    dependency=(--dependency="afterok:${previous}")
  fi
  treatment_job="$(
    CONFIG="${config}" EVAL_MODULE=math_eval RUN_NAME="${treatment_run}" \
    METHOD="${method}" CHECKPOINT=step-200 ADAPTER="${training_dir}" \
    sbatch --parsable \
      "${dependency[@]}" \
      --nodelist=node10 --gres=gpu:8 --ntasks=8 --cpus-per-task=8 --mem=900G \
      --output="${SCRATCH_ROOT}/slurm-opsd-eval-%j.out" \
      infra/turing/run_eval.sbatch
  )"
  baseline_job="$(
    CONFIG="${config}" EVAL_MODULE=math_eval RUN_NAME="${baseline_run}" \
    METHOD=untouched CHECKPOINT=none \
    sbatch --parsable \
      --dependency="afterok:${treatment_job}" \
      --nodelist=node10 --gres=gpu:8 --ntasks=8 --cpus-per-task=8 --mem=900G \
      --output="${SCRATCH_ROOT}/slurm-opsd-eval-%j.out" \
      infra/turing/run_eval.sbatch
  )"
  printf 'ch1_full_%s_treatment=%s run=%s\n' \
    "${benchmark}" "${treatment_job}" "${treatment_run}"
  printf 'ch1_full_%s_baseline=%s run=%s\n' \
    "${benchmark}" "${baseline_job}" "${baseline_run}"
  locked_runs+=("${treatment_run}" "${baseline_run}")
  previous="${baseline_job}"
done

locked_controller="$(
  sbatch --parsable \
    --dependency="afterok:${previous}" \
    --output="${SCRATCH_ROOT}/slurm-after-ch-full-locked-%j.out" \
    --export="ALL,CH_AIME25_TREATMENT_RUN=${locked_runs[0]},CH_AIME25_BASELINE_RUN=${locked_runs[1]},CH_AIME26_TREATMENT_RUN=${locked_runs[2]},CH_AIME26_BASELINE_RUN=${locked_runs[3]}" \
    reproductions/06_graf_opsd/after-ch-full-locked.sbatch
)"
echo "ch1_full_locked_comparison=${locked_controller}"
