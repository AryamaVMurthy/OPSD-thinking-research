# FiNOD execution plan

## Registered hypothesis

An answer-masked procedural guide contains both useful procedural steering and
answer-control-aligned steering. At student rollout prefixes, removing only
positive Fisher alignment with a matched destination-only control should
retain a nonzero, reachable procedural target and avoid the trajectory damage
seen in the previous full-context teacher objectives.

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
