# Fluid G4: compute-efficient GRAF-OPSD design

Status: approved for bounded development on 2026-07-27.

## Objective

Turn the positive but underpowered G4 pilot into a numerically correct,
coverage-preserving, free-form reasoning method. The development campaign is
capped at 60 A100 GPU-hours before a new scale decision. The legacy automatic
controller job 16394 remains held.

The method must preserve these invariants:

- the student sees only the problem and its own on-policy prefix;
- privileged answers, references, critiques, and viability outcomes are
  teacher-only;
- model-generated mathematical reasoning is natural text, not a required JSON
  schema, named-method list, fixed mistake taxonomy, or fixed branch count;
- JSON is permitted only as an internal cache/checksum envelope;
- graph or action coverage must not reduce the ordinary OPSD training stream;
- AIME 2025/2026 are not development-selection data for the new method.

## Evidence motivating the redesign

The completed G4 run trained 50 optimizer steps on only 22 active identities.
The source cache requested 128 examples, accepted 62 graphs, and produced 171
forks, but 142 forks had uniform targets. Two viability samples per action
made estimates take only 0, 0.5, or 1. The launcher then retained only examples
with informative forks, so all ordinary OPSD updates also recycled those 22
identities.

Job 16356 completed stably at the optimizer level, but 1,080/1,200 rollout
calls reached 4,096 tokens. Only 70/300 retained rollouts closed thinking.
The branch KL decreased over training, but its BF16 reduction occasionally
reported small negative values. The configured information weight usually
cancels during normalization when a batch has one active fork, and the
entropy-floor term became effectively zero late in training.

The existing token objective is a forward KL with each vocabulary contribution
clamped above at 0.05. This clipped sum is not a canonical divergence and is
not guaranteed non-negative. The G4 teacher receives an answer-masked scaffold
rather than the complete verified solution and comparative mistake analysis.

## Method

For problem \(x\), the student policy \(S_\theta\) samples an answer-blind
trajectory. A fixed teacher \(T\) receives a teacher-only evidence packet:

1. the verified answer and reference reasoning;
2. two or three independent answer-blind student attempts;
3. an unrestricted natural-language comparison of successful, failed, and
   recoverable reasoning;
4. a variable set of short natural continuation proposals;
5. empirical continuation outcomes with uncertainty.

The evidence packet is rendered as prose. Storage may have typed fields for
provenance, but no model response is admitted or rejected based on a reasoning
schema.

Every problem remains eligible for the base on-policy distillation loss.
Action supervision is a sparse auxiliary loss: examples without a reliable
action comparison receive ordinary OPSD rather than being removed.

For each candidate action, collect two forced continuations initially. Use a
Beta-binomial posterior and allocate at most two additional continuations only
when credible intervals overlap enough to leave the action ordering uncertain.
Invalid actions can retain zero target mass, but measured non-invalid actions
carry posterior uncertainty rather than false point certainty.

The auxiliary coefficient is based on absolute evidence strength and must not
be normalized away when only one fork is active. Entropy preservation is
enabled only where at least two actions retain credible success mass.

## Canonical token objectives

Let \(T\) and \(S\) denote teacher and student vocabulary distributions at the
same student-generated prefix.

\[
L_\mathrm{FKL} = \mathrm{KL}(T\|S)
\]

\[
L_\mathrm{RKL} = \mathrm{KL}(S\|T)
\]

\[
M = \tfrac12(T+S), \qquad
L_\mathrm{JS} = \tfrac12\mathrm{KL}(T\|M)
              + \tfrac12\mathrm{KL}(S\|M).
\]

All log-softmax and vocabulary reductions run in FP32 for BF16 model logits
(FP64 is preserved for reference callers). Vocabulary chunking may alter only
reduction order: it must not truncate, renormalize, use Top-K, or clip
individual vocabulary contributions.

The primary divergence ablation changes only this token objective. The action
loss stays forward KL during that ablation. Reverse KL is not applied to an
action target containing hard zeros without a separately registered smoothing
protocol.

### Public interface

Expose one function:

```python
exact_divergence_vocab_chunked(
    student_logits,
    teacher_logits,
    labels=None,
    *,
    divergence="forward_kl",
    temperature=1.0,
    reduction="batchmean",
    chunk_size,
)
```

Supported divergence names are exactly `forward_kl`, `reverse_kl`, and `js`.
The historical `exact_forward_kl_vocab_chunked` name remains as a compatibility
wrapper during migration.

Required behavioral tests:

- analytic agreement with PyTorch/reference formulas;
- zero at equal distributions;
- non-negativity within stated floating-point tolerance;
- correct label masking and denominator;
- chunked versus unchunked agreement;
- finite, correct-direction student gradients;
- teacher gradients are not introduced by the training call site;
- invalid shapes, temperatures, reductions, and divergence names fail fast;
- FP32 work reduction for BF16/FP16 inputs.

## Stability telemetry

Every optimizer step records:

- optimized objective and unoptimized FKL/RKL/JS diagnostics;
- mean, p50, p90, p99, minimum, maximum, and non-finite counts;
- student and teacher entropy and their gap;
- gradient norm, clipping occurrence, LoRA parameter/update norm, and LR;
- branch KL, entropy term, absolute confidence mass, active forks, actions per
  fork, and posterior uncertainty;
- identity count, repeat count, coverage, sampling Gini, and active-branch rate;
- rollout mean/p90 length, thinking closure, boxed-answer rate, cap rate;
- generation time, training time, tokens/s, GPU utilization, peak memory, and
  cumulative GPU-hours.

Canonical divergences are never silently repaired. A materially negative or
non-finite value fails the experiment.

## Development ladder and stopping rules

### Stage A: CPU correctness

Implement the public objective and telemetry test-first. Run the targeted tests
and full local suite. GPU cost: zero.

### Stage B: fixed-cache numerical ablation

Use the immutable G4 cache only for controlled plumbing/stability comparisons:

1. one-step FKL, RKL, and JS smokes;
2. five-step pilots for the two numerically healthiest objectives.

A smoke cannot establish task quality. Reject only for incorrect loss,
non-finite gradients, excessive memory, or a throughput regression above 15%
without a compensating correctness reason. Budget: 8--12 A100 GPU-hours.

### Stage C: fluid-cache pilot

Build 256--512 source identities with variable natural proposals and adaptive
viability sampling. Require at least 95% complete evidence-packet coverage and
retain 100% of otherwise eligible identities in the base OPSD stream. Cache
budget: at most 3 A100 GPU-hours.

### Stage D: successive-halving training

Train the divergence winner and runner-up for 12--25 steps with identical data,
rollout seeds, and maximum generated-token budgets. Use two training seeds for
the final development comparison. Evaluate paired AIME 2024 Average@4 and one
non-AIME held-out mathematical set. Budget: 25--40 A100 GPU-hours including
development evaluation.

Promote only if:

- both seeds have finite, well-behaved optimization;
- the aggregate paired correctness movement is positive;
- Pass@k and majority accuracy do not show a systematic regression;
- identity coverage is materially broader than G4's 22 examples;
- completed correct trajectories per GPU-hour improve;
- graph/action supervision beats the matched plain-OPSD control.

### Stage E: scale decision

The 60-GPU-hour cap ends at the promotion decision. A promoted method first
receives a 1,600-identity/50-step confirmation. Publication-scale
6,000+-identity, 200-step, multi-seed training and the full locked benchmark
suite require a separately reported compute estimate and explicit continuation
decision.

## Experiment ledger

The existing session TSV remains the append-only campaign ledger. Each GPU
hypothesis receives its own `autoresearch/2026-07-27-fluid-g4/<experiment>`
branch and commit, with parent/commit hashes, immutable config, cache hashes,
job ID, exact command, runtime, GPU-hours, metrics, and keep/discard/crash
decision. Failed branches are preserved.

The authoritative baseline is completed G4 job 16356 plus its checksum-verified
evaluations. No new GPU run begins until the CPU objective implementation,
tests, config validation, source commit, and artifact paths are recorded.
