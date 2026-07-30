# FiNOD execution plan

## Registered hypothesis

A problem-only procedural guide can induce both useful procedural steering and
destination-like steering after it is inserted into the frozen teacher. At
student rollout prefixes, removing only positive Fisher alignment with a
matched destination-only control should retain a nonzero, reachable procedural
target and avoid the trajectory damage seen in the previous full-context
teacher objectives.

The first candidate is fixed before GPU inspection:

- Qwen3-4B instruction checkpoint;
- the immutable accepted G1 answer-masked scaffolds;
- current-student rollouts, 2,048-token completion cap;
- 32 uniformly spaced valid rollout positions;
- one-sided Fisher answer-control projection;
- exponential-tilt step size 0.25;
- exact per-token target-KL cap 0.01;
- LoRA rank 64 on all projection modules;
- AdamW through the pinned upstream Trainer, learning rate `5e-6`;
- global gradient clipping at `0.1`;
- DP8, gradient accumulation 4, effective batch 32.

## Answer-blind protocol correction

The first five-step optimization probe used historical G1 scaffolds generated
with access to a reference solution. A subsequent semantic audit found an
accepted guide containing `4/3` when the boxed reference used
`\frac{4}{3}`. Consequently, that probe and its paired AIME-2024 evaluation
are behavioral diagnostics only; they cannot support an answer-free claim.

Every cache used for the representative run is instead built under
`problem-only-v1`: the guide generator receives the problem and no reference
answer or solution. Privileged teacher critique is forbidden. The reference is
used only after generation by a rejecting audit. The audit canonicalizes
common LaTeX, Unicode, fraction, decimal, and percent aliases; rejects asserted
numerical results; verifies the immutable cache, graph, and problem hashes;
and replays every accepted scaffold through the exact training-row adapter.
The cache manifest and every record must declare
`surface-equivalence-and-result-claim-v2`, and the representative launcher
refuses any other provenance.

This construction establishes that no privileged reference entered the guide
generator. It does not claim that a model shown the problem cannot itself
derive the answer; that would be neither possible nor desirable.

## Stage 1: one-step engineering smoke

The smoke is valid only if all eight A100s are bound, the source tree and
manifests are checksum recorded, one optimizer step completes, the adapter and
rollout dump are saved, and no CUDA/Python/data error occurs.

## Stage 2: five-step mathematical signal screen

Proceed only if:

- every loss and gradient norm is finite;
- the unrounded FiNOD loss is positive;
- mean residual Fisher energy is positive;
- fewer than 80% of selected tokens collapse below residual energy `1e-8`;
- positive post-projection nuisance alignment is numerically negligible;
- measured target KL never exceeds `0.01` beyond floating-point tolerance;
- the realized adapter update is nonzero and bounded;
- student prompts contain neither the guide nor answer-control fields.

If the residual collapses, first lower the nuisance model's scope rather than
increasing the learning rate. If the residual is healthy but optimization is
too weak, adjust the target-KL radius or number of selected positions one at a
time. If optimization is healthy but accuracy degrades, test a smaller
target-KL radius before adding new components.

## Stage 3: development accuracy screen

Use a fixed representative AIME-2024 development set, six paired rollouts per
problem, full thinking context, and a 32,768-token generation cap. Compare
sample accuracy, majority accuracy, pass rate, output length, format rate, and
per-problem paired changes against the untouched checkpoint. This is a method
screen, not a paper result.

## Stage 4: larger validation

Only a positive development signal advances to a cache of at least 1,024
source problems stratified by available mathematical topic metadata and
length/difficulty proxies. Preserve six rollouts per evaluation problem.
Locked AIME-2025 and AIME-2026 are evaluated once, with the same 32,768-token
budget, only after the larger run and an independent AIME-2024 confirmation.

The registered representative run uses exactly 1,024 accepted, non-held-out
records and 32 optimizer steps at effective batch size 32, giving one complete
pass without replacement. Selection is deterministic and stratified jointly
by the dataset's native `data_source` family and response-length quartile. Its
source indices, per-stratum counts, and checksum are saved with the training
artifacts. The rollout cap is 4,096 because 159 of 160 five-step-screen
rollouts reached the original 2,048-token cap.
