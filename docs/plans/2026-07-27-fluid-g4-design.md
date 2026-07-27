# Fluid G4: compute-efficient GRAF-OPSD design

Status: approved for bounded development on 2026-07-27.

## Objective

Turn the positive but underpowered G4 pilot into a numerically correct,
coverage-preserving, free-form reasoning method. The development campaign is
capped at 60 allocated GPU-hours before a new scale decision. Development is
currently running on four RTX 6000 Ada GPUs, so accounting must name those
devices and must not relabel their hours as A100-hours. The legacy automatic
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
without a compensating correctness reason. Budget: 8--12 allocated GPU-hours.

### Stage C: fluid-cache pilot

First reuse the checksum-verified 128-packet cache for an engineering smoke and
short promotion pilot. It already contains the required free-form blind
attempts, verified reference, audit, and measured continuations, so rebuilding
it would spend compute without changing the hypothesis. Expand to 256--512
source identities only after the 128-packet run passes its numerical and
matched-quality gates. Require at least 95% complete evidence-packet coverage
and retain 100% of otherwise eligible identities in the base OPSD stream.
Expanded-cache budget: at most 3 allocated GPU-hours.

### Stage D: successive-halving training

Train the divergence winner and runner-up for 12--25 steps with identical data,
rollout seeds, and maximum generated-token budgets. Use two training seeds for
the final development comparison. Evaluate paired AIME 2024 Average@4 and one
non-AIME held-out mathematical set. Budget: 25--40 allocated GPU-hours including
development evaluation, measured on the named hardware.

Promote only if:

- both seeds have finite, well-behaved optimization;
- the aggregate paired correctness movement is positive;
- Pass@k and majority accuracy do not show a systematic regression;
- identity coverage is materially broader than G4's 22 examples;
- completed correct trajectories per GPU-hour improve;
- graph/action supervision beats the matched plain-OPSD control.

The non-AIME set is also the early detector for valid-prefix overwrite.
Contemporary reflective OPSD reports that full-response distillation may
damage already-correct prefixes and out-of-domain generalization. We therefore
do not select a checkpoint from AIME alone. A soft-localization experiment is
permitted only when the unmasked method has a positive in-domain signal but a
repeatable transfer regression; introducing it earlier would add another
factor before establishing that the evidence packet works.

### Stage E: scale decision

The 60-GPU-hour cap ends at the promotion decision. A promoted method first
receives a 1,600-identity/50-step confirmation. Publication-scale
6,000+-identity, 200-step, multi-seed training and the full locked benchmark
suite require a separately reported compute estimate and explicit continuation
decision.

Before publication-scale training, measure evidence staleness on a fixed
diagnostic subset. If current action preferences or continuation outcomes have
drifted materially, compare a token-budget-matched selective-refresh schedule
against the fixed cache. A fixed-versus-periodically-synchronized teacher
ablation is required only for the longer run, where teacher lag can become
material; it is not added to the 12-step screening factorial.

## Compute-efficient execution amendment

The shortest valid path is:

1. use one optimizer step only to reject broken FKL/RKL/JS implementations;
2. run five steps only for the two implementations that pass correctness,
   memory, and throughput gates;
3. run one fluid-FKL engineering step on all 124 training-eligible dossier
   identities, with the 22 informative identities receiving the sparse
   auxiliary;
4. promote at most two arms to 12 steps;
5. use paired AIME 2024 Average@4 plus a non-AIME held-out set to choose one;
6. if and only if the effect is positive, run the matched 124-row JS dossier
   control (same blind attempts, verified solution, audit, seed, and token
   budget; no empirical graph outcomes or route loss);
7. then separate teacher-visible outcome context from the action auxiliary,
   and compare a matched reference-only OPSD control, before expanding the
   cache or training longer.

Interpret the context-only control asymmetrically. If it matches full
Fluid-G4 within paired uncertainty, remove the action forward from the
compute-efficient method. If it clearly beats full Fluid-G4, test exactly one
repair: replace the value-only Boltzmann target with a detached
behavior-prior, mirror-descent target at the same continuation and token
budget. If full Fluid-G4 wins, retain the simpler registered target. This
prevents an open-ended action-loss search.

The divergence ablation changes only FKL versus RKL versus JS. Evidence-packet
components, action target, seeds, token budget, optimizer, and checkpoint are
held fixed. Packet-component ablations are postponed until there is a positive
fluid result; otherwise they cannot explain an effect that does not exist.

Early stopping is asymmetric. Numerical invalidity, OOM, missing coverage,
leakage, or a material throughput regression rejects immediately. A noisy loss
increase over one to five updates does not reject by itself; the decision uses
finite gradients, pre-clip exceedance rate, realized adapter-update norm,
student/teacher entropy gap, branch/base balance, cap rate, and paired task
quality.

As of the amendment, exact FKL and RKL each passed one update. Ordinary JS
exceeded memory before its first update, which rejects that implementation but
not JS; a mathematically equivalent recomputed-backward JS is the only repair
arm. The five-step FKL stability run is the current reference. No three-way
long pilot will be run merely for completeness.

## Experiment ledger

The existing session TSV remains the append-only campaign ledger. Each GPU
hypothesis receives its own `autoresearch/2026-07-27-fluid-g4/<experiment>`
branch and commit, with parent/commit hashes, immutable config, cache hashes,
job ID, exact command, runtime, GPU-hours, metrics, and keep/discard/crash
decision. Failed branches are preserved.

The authoritative baseline is completed G4 job 16356 plus its checksum-verified
evaluations. No new GPU run begins until the CPU objective implementation,
tests, config validation, source commit, and artifact paths are recorded.

## Revised causal ladder after the first quality result

The registered 12-step Fluid-G4/JS arm was stable but failed its AIME-2024
development gate: Average@4 changed from 77.50% to 74.17%, and Pass@4 changed
from 86.67% to 80.00%. Seven of eight degraded paired samples came from
problems the base model solved 4/4. The full method is therefore rejected; its
stable loss is not evidence of useful learning.

The shortest causal ladder is now:

| Gate | One factor changed | Maximum work before a task signal | Decision |
|---|---|---:|---|
| A0 | Remove action scoring and route loss, retain the complete empirical teacher packet | 12 steps + paired AIME-2024 Average@4 | Attribute damage to the route auxiliary or to full-response teacher matching |
| A1 | Add verifier-based outcome protection to the better context formulation | 5 steps, then 12 only if stable | Reject if it does not recover correct-to-wrong transitions and paired accuracy |
| A2 | Change the distillation horizon from all 4,096 tokens to the first 1,024 tokens | 5 steps, then one paired screen only if A1 is positive | Test whether long-horizon imitation causes search/verbosity damage |
| A3 | Replace uniform token weights with a soft entropy/disagreement gate | one 5-step arm; paired screen only if A2 leaves a measured localization problem | Compare against TSD-KD-style selection; do not claim gating itself as novel |
| A4 | Hold the winning method fixed and change JS to exact FKL or exact RKL | 5 steps per alternative; promote at most one | Complete the divergence ablation without a three-way long factorial |
| A5 | Repeat the winner with seed 43 and evaluate AIME-2024 plus HMMT-Feb-2025 | two matched 12-step runs/evaluations | Require directionally consistent task movement and no systematic Pass@k or transfer loss |
| A6 | Expand identities and train 50 steps | only after A5 | Measure scaling, cache staleness, and correct trajectories per GPU-hour |

Job 16627 is A0. Its configuration differs from full Fluid-G4 only by
disabling action routing. The dependent controller validates all 12 updates
and automatically compares the same paired evaluation with both the immutable
base and full Fluid-G4. AIME 2025/2026 remain locked.

### Outcome-protected fluid objective

If A0 remains below the base, A1 uses a task-level verifier score
\(r(y)\in[0,1]\) and a continuous protection weight \(w_\mathrm{fail}(y)\).
Incorrect, incomplete, or low-confidence trajectories receive privileged
teacher distillation. Verified-correct trajectories are protected from
full-response imitation; a short positive tail may receive ordinary
student-target SFT as a separately logged anchor. This is an established
outcome-selection repair, not the novelty claim.

The teacher packet remains unrestricted prose. It may contain independent
answer-blind attempts, a verified reference, comparative audit, and empirical
continuation outcomes, but the student never sees those fields. The training
interface accepts a general scalar verifier rather than an AIME-specific
answer rule, making the same objective applicable to tests for code, proof
checking, or execution success for tool use. JSON remains provenance storage,
not a required reasoning format.

For a student token \(t\), the candidate objective is

\[
L_t =
w_\mathrm{fail}(y)\,
w_\mathrm{position}(t,y)\,
D\!\left(T_t(\cdot\mid x,z,y_{<t}),S_t(\cdot\mid x,y_{<t})\right),
\]

with \(D\) fixed to exact JS for the first repair. Outcome selection,
distillation horizon, token localization, and divergence are changed in
separate gates. This prevents a positive result from being uninterpretable.

The implementation contract is deliberately task-general:

1. each training row carries a verifier target in a separate, teacher-hidden
   metadata field derived from the immutable source/cache record; it is never
   recovered by parsing the free-form dossier;
2. after the live student rollout, a verifier returns a score in \([0,1]\),
   confidence, and an auditable status such as verified, contradicted,
   incomplete, or unavailable;
3. only the scalar score and confidence affect the loss; the status is
   telemetry and must not become a required model-output schema;
4. response weights are synchronized across data-parallel ranks and
   normalized by their detached global mean, with an explicit zero-active
   path, so the learning rate does not fluctuate merely because one rank
   sampled a correct response;
5. padding, horizon, and position masks are applied before the token
   denominator is computed. A batch with no active tokens returns a
   differentiable zero rather than dividing by zero;
6. the exact same response weights and active-token denominator are used by
   FKL, RKL, and JS.

For the current math study, the target is the cache's separately stored
`reference_answer` and the verifier is the pinned MathArena-equivalence
checker. For code it can be a test pass fraction; for proofs, checker
acceptance; for tool use, execution success. This introduces no additional
teacher forward. Verifier latency, unavailable/error rate, active response
fraction, and global normalization factor are logged every optimizer update.

The 128-row cache audit found that the separately stored answer exists for all
128 identities and never uses the dossier builder's fallback text. However,
the pinned MathArena checker accepts only 124/128 answers when each canonical
answer is compared with itself. Its four failures cover an inequality, the
scalar zero, a three-element list, and an equation. Therefore the official
checker cannot be the sole online training verifier. Before A1 is launched:

- every target must pass a canonical self-check;
- an independently implemented math-equivalence checker plus normalized exact
  answer match must be evaluated as fallbacks;
- disagreement or checker exception must yield `unavailable`, never
  `incorrect`;
- outcome weighting must report coverage by verifier route and retain
  ordinary base OPSD for unavailable rows;
- the paired benchmark continues to use the official MathArena sidecar,
  independently of the training verifier.

This separates a training signal-quality problem from benchmark reporting and
prevents parser failures from creating false-negative distillation pressure.

A second canonical audit found that the repository's `math_verify` helper
accepts only 89/128 reference answers when each target is compared with
itself. On the 256 cached blind attempts, that helper labels 63 correct while
the official parser labels 79; they agree on 61 correct attempts, disagree on
20, and agree on 175 negatives. Neither implementation can be treated as an
oracle.

This also invalidates a stronger interpretation of the historical action
cache. Its 790 forced-continuation labels were produced by the same
`math_verify` helper, and only aggregate success booleans were retained. Raw
completion text, extracted answer, parser status, and finish reason were not
stored, so the labels cannot be regraded. The cache remains valid only as an
immutable pilot artifact; it is not publication-quality empirical outcome
evidence.

Consequently the post-A0 decision tree is refined:

1. if context-only is positive, replicate with a newly auditable outcome
   cache before attributing the gain to empirical continuations;
2. if context-only is negative, run the already-preregistered dossier-only
   control, which removes empirical continuation prose as well as routing;
3. if dossier-only recovers, rebuild outcome evidence and do not add a new
   loss yet;
4. only if dossier-only also damages quality is verifier-based response
   protection the next causal repair.

Every new forced continuation record must retain immutable problem/action
identity, prompt hash, generation seed, raw completion, output token IDs or
their checksum, finish reason, extracted answer, each verifier's structured
result, ensemble decision, and policy version. Aggregate Beta counts are
derived artifacts, never the only stored evidence.

A last-128-token positive-tail SFT anchor for verified-correct short
trajectories is a separate switch because it changes the objective. It is an
OGLS-SD baseline, not bundled into the first outcome-gating test. If used, its
coefficient, active-token count, and gradient contribution are reported
separately from distillation.

### Stability and mechanism acceptance criteria

Every promoted screen must report, over the entire run rather than only the
last batch:

- loss, pre-clip gradient norm, clip rate, learning rate, adapter parameter
  norm, update norm, and update/parameter ratio;
- exact FKL, RKL, and JS token statistics, non-finite/material-negative
  counts, student/teacher entropy, and normalized position quartiles;
- rollout length distribution, 4,096-token cap rate, thinking closure,
  extracted-answer rate, correctness, and correct answers per generated
  million tokens;
- paired correct-to-wrong and wrong-to-correct transitions, stratified by base
  0/4 through 4/4 performance;
- teacher answer consistency, evidence coverage, identity repeats, sampling
  concentration, and—when enabled—route/base loss balance;
- wall time, generated tokens, peak memory, utilization, cache cost, training
  GPU-hours, evaluation GPU-hours, and teacher-forward equivalents.

Immediate rejection conditions are leakage, missing coverage, non-finite
optimization, materially negative canonical divergence, OOM, or a persistent
unexplained throughput regression above 15%. Task rejection is based on paired
accuracy and transition structure, not on a smooth training loss.

### Compute envelope on the current four-GPU node

Observed Fluid-G4 costs provide the planning unit:

- 12-step training: 6.61 allocated RTX 6000 Ada GPU-hours;
- paired AIME-2024 Average@4 evaluation: 2.05 allocated GPU-hours;
- one complete train-and-screen arm: about 8.7 allocated GPU-hours;
- one five-step numerical screen: about 2.8 allocated GPU-hours.

Consequently A1 first spends about 2.8 GPU-hours, not 8.7. Only a stable A1
spends the additional 12-step screen. A2 and A3 are conditional, not queued as
a grid. FKL and RKL together cost about 5.6 GPU-hours at five steps; at most
one receives a task evaluation. This ordering limits the remaining
method-selection campaign to roughly 20--35 allocated GPU-hours if early
gates work, while negative gates stop much earlier. Evidence already cached is
reused and reported as sunk compute; it is not silently excluded from the
final efficiency table.

### Eight-GPU execution gate

Turing node10 currently exposes eight A100 GPUs, 256 CPUs, and approximately
1 TB host memory. Historical node10 telemetry is consistent with 40 GB A100s,
not the 48 GB RTX 6000 Ada cards used by the active Fluid-G4 runs. Therefore
availability does not by itself authorize an eight-GPU long run.

The next positive repaired-JS candidate first receives one optimizer step on
node10 with data parallelism 8 and gradient accumulation 4, preserving
effective batch size 32. It must reproduce the four-GPU loss within
rollout-induced variation, save an adapter, and stay below a 39.5 GiB
per-device memory guard. Exact recomputed JS used about 38.1 GiB on the Ada
node and is the only current objective with a plausible margin. The older
ordinary-RKL path used about 45.2 GiB and cannot be assumed to fit; every
divergence receives its own memory smoke.

Only after the smoke passes does node10 become the default for subsequent
training, with all eight GPUs used and the same effective batch, data seed,
rollout seed, and token budget. Speedup is reported from measured
examples/second and wall time rather than assumed to be 2x. Node-local
scratch means adapters and result sidecars must be checksum-staged when a
controller changes nodes; a live node03 job is never restarted merely to
chase temporary node10 availability.
