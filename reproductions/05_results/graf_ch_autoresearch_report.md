# GRAF / dynamic contrastive-hindsight OPSD autoresearch

Status: completed development phase, stopped at the preregistered promotion
gate on 2026-07-27. This report records only completed, checksum-verified
results. No tested GRAF/CH candidate cleared the gate for paper-scale training,
so the 6,912-row cache, 200-step run, and locked AIME25/AIME26 evaluations were
not launched.

## Scientific question

Can thinking-enabled Qwen3-4B learn better mathematical reasoning from a fixed
privileged teacher while keeping the on-policy student answer-blind, preserving
free-form reasoning, and avoiding the cost and instability of a large
parameter sweep?

The model, model revision, Math-CoT-20k revision, optimizer, effective batch,
LoRA configuration, sampling parameters, AIME evaluation seeds, and scorer are
pinned. AIME24 is development data. AIME25 and AIME26 remain locked until a
fresh paper-scale AIME24 result is positive.

## Methods evaluated

### G4: viability-routed GRAF

G4 uses graph-derived route/recovery targets and 4,096-token thinking rollouts.
Its 50-step run was an engineering and development pilot, not a paper-scale
training run: only 22 unique routed training identities were available.

### CH: dynamic contrastive-hindsight OPSD

For each Math-CoT problem:

1. The student receives only the problem and independently samples natural
   solutions.
2. A privileged auditor receives the problem, trusted answer/reference, and
   blind attempts, then writes unrestricted comparative prose.
3. The fixed teacher receives the complete natural dossier when scoring a new
   on-policy student trajectory.
4. The on-policy student still receives only the problem.

The production path does not require JSON, named methods, a fixed critique
ontology, or schema-shaped mathematical reasoning. Examples are gated only by
source validity, generation success, cache coverage, and exact context fit.
Long examples are accommodated by context budgeting rather than silently
filtered.

CH0 uses three 2,048-token blind attempts and 2,048-token training rollouts.
CH1 uses two 4,096-token blind attempts and 4,096-token training rollouts.
CH1's 12-step pilot permits at most 1,572,864 rollout tokens, about 4% fewer
than CH0's 1,638,400-token 25-step pilot. This isolates completion allocation
instead of rewarding CH1 with more training tokens.

## Completed G4 results

| Result | Value |
|---|---:|
| Training job | 16356 |
| Training steps / GPUs | 50 / 4 |
| Training wall time | 5:04:44 |
| Unique routed identities | 22 |
| Official AIME24 Avg@12 | 0.775000 |
| Official AIME24 Pass@12 | 0.866667 |
| Official AIME24 Maj@12 | 0.833333 |

The forced 4,096-token paired screen gave G4 minus untouched Avg@4 =
-0.066667, but almost every response was length-truncated. It is retained only
as conciseness evidence.

The valid full-context paired screen gave:

| Full-context AIME24 Avg@4 | Value |
|---|---:|
| Untouched Qwen3-4B | 0.775000 |
| G4 | 0.783333 |
| Paired delta | +0.008333 |
| Paired bootstrap 95% CI | [-0.058333, +0.066667] |

G4 is viable but unproven: its interval crosses zero and its 22-identity
training set is not publication-scale.

## CH cache experiments

| Cache | Coverage | Main result | Decision |
|---|---:|---|---|
| 16412, 3 attempts, 16,384 context | 118/128 | 10 long prompts overflowed | Reject; increase context, do not filter |
| 16418, 3×1,024 | 128/128 | Every blind attempt hit the cap | Reject as unfinished reasoning |
| 16422, CH0 3×2,048 | 128/128 | Any correct 9.375%; ~98–99% cutoff | Train CH0; diagnose truncation |
| 16444, CH1 2×4,096 | 128/128 | Any correct 32.031%; ~80% cutoff | Keep as completion-aware recovery |

CH0 cache 16422 had a maximum exact teacher prompt of 24,310 tokens under the
28,672-token training budget. Its third blind attempt uniquely recovered only
3/128 answers (2.34375%), motivating two longer attempts rather than three
short attempts.

CH1 cache 16444 completed in 6:39 on four independent GPU replicas:

| CH1 cache metric | Value |
|---|---:|
| Requested / accepted | 128 / 128 |
| Attempt correctness | 0.234375 / 0.257812 |
| Any-attempt correctness | 0.320312 |
| Blind cutoff rate | 0.804688 / 0.796875 |
| Audit cutoff rate | 0.015625 |
| Maximum teacher prompt | 26,781 / 28,672 |
| Total generated cache tokens | 1,101,488 |

The improvement from 9.375% to 32.031% any-attempt correctness is direct
evidence that 2,048 tokens prevented many sampled methods from reaching an
answer.

## CH0 training and evaluation

CH0 smoke job 16424 completed five finite steps in 23:13 with approximately
34 GiB peak memory per GPU. Pilot 16426 completed 25 finite optimizer steps in
1:45:22. Of 775 logged training rollouts, 761 reached the 2,048-token cap.

Full-context paired AIME24 result:

| Avg@4 | Value |
|---|---:|
| Untouched | 0.775000 |
| CH0 | 0.758333 |
| Paired delta | -0.016667 |
| Paired bootstrap 95% CI | [-0.100000, +0.066667] |
| Majority delta | 0.000000 |
| Pass delta | -0.033333 |

CH0 is not promoted. The change was concentrated in a few problems rather
than a broad collapse, but its near-universal rollout truncation prevents it
from being the final method.

## CH1 training and evaluation

CH1 cache 16444 is immutable and checksum-verified. A one-step engineering
smoke completed generation, backward, and optimization without OOM:

| Smoke metric | Value |
|---|---:|
| Loss | 0.0004 |
| Gradient norm | 0.0210 |
| Peak observed memory | ~40.7 / 46.1 GiB per GPU |
| GPU utilization | ~96–97% |

Pilot 16460 completed 12/12 finite steps in 1:39:16. The final root adapter
and checkpoint-12 adapter have identical SHA-256
`f858993e2a52c73af8e6dffe2d650bba64e246754e8c666968b2cf988675234e`.
The final summary verifies checkpoint and rollout coverage through step 12.

| Pilot metric | Value |
|---|---:|
| Final loss | -0.0014 |
| Final gradient norm | 0.01854 |
| Observed nonempty rollouts | 93 / 93 |
| Generation calls at 4,096-token cap | 311 / 372 |
| Mean GPU utilization | 94.45–94.95% |

Full-context evaluation 16465 completed in 30:21, and every artifact passed
its checksum manifest. Only 1/120 responses reached the 38,912-token
evaluation limit.

| Paired AIME24 Avg@4 | Value |
|---|---:|
| Untouched | 0.775000 |
| CH1 | 0.775000 |
| Paired delta | 0.000000 |
| Paired bootstrap 95% CI | [-0.083333, +0.075000] |
| Majority delta | 0.000000 |
| Pass delta | -0.033333 |

There were seven improved and seven degraded paired samples. Changes were
localized rather than uniform: CH1 gained two samples on each of AIME I:8 and
II:11, but lost three on I:13. Only one degraded sample was evaluation-length
truncated, so the neutral result cannot be attributed to the evaluation cap.
CH1 did not advance directly to the 6,912-row/200-step paper-scale run. Its
exactly neutral result admitted only the single preregistered same-method
diversity/dose confirmation below.

## CH1 diversity/dose confirmation

The single allowed same-method confirmation is complete. It changed only the
number of unique training identities and optimizer steps; the CH1 free-form
dossier method, fixed teacher, loss, sampling, and evaluation gate remained
unchanged.

Eight-GPU cache job 16473 completed in 25:15. All artifacts passed their
SHA-256 manifest.

| Confirmation-cache metric | Value |
|---|---:|
| Requested / accepted / rejected | 1,682 / 1,682 / 0 |
| Eligible first-pass training identities | 1,600 |
| Blind attempts per identity | 2 |
| Any blind attempt correct | 30.737% |
| Blind cutoff rate, attempts 1 / 2 | 76.100% / 76.813% |
| Mean blind tokens, attempts 1 / 2 | 3,828.3 / 3,824.6 |
| Mean auditor output tokens | 744.4 |
| Mean / maximum teacher prompt tokens | 14,859 / 26,784 |
| Schema-based selection | Disabled |
| Acceptance rate | 100% |

The 40 GiB A100 confirmation required a bounded operational memory study
before step 1. These attempts did not write a checkpoint or accept an
optimizer step:

| Job / setting | Observed result |
|---|---|
| 16477, vLLM 0.35 | Exact-KL backward OOM; 2.32 GiB request with ~2.0 GiB free |
| 16483, vLLM 0.35 + expandable segments | Peak reached ~40.4 GiB; cuBLAS handle allocation failed |
| 16486, vLLM 0.25 + expandable segments | vLLM rejected initialization: 1.72 GiB KV available versus 3.94 GiB required |
| 16491, vLLM 0.32 + expandable segments | Full 28,672 context fits; checkpoint 25 validated at ~180 seconds/step |

Job 16491 preserves the effective batch of 32, 4,096-token completion limit,
28,672-token context, exact full-vocabulary forward KL, and all 1,600
first-pass identities. It completed 50/50 finite steps in 2:31:53 with exit
code 0, exactly one epoch over the 1,600 identities. Final trainer state has
50 log rows; checkpoint 50 is 2.8 GiB and contains the LoRA adapter, trainer
and scheduler state, eight RNG states, eight DeepSpeed optimizer shards, and
the model state. The final rollout dump covers checkpoint 50, and all 200
recorded rollouts across the retained five-step dumps are nonempty.

| Confirmation-training metric | Value |
|---|---:|
| GPUs / optimizer steps | 8 / 50 |
| Wall time / mean step time | 2:31:53 / 180.0 s |
| Mean / final logged loss | -0.004466 / -0.0072 |
| Final gradient norm | 0.01960 |
| Root adapter SHA-256 | `d36284b7a411f32cb11da33c50957897f4107c2fe97ce274f534e27e2ab8f540` |
| Final generation-dump SHA-256 | `71ebe2f6e00ce445641aaff3cba4e815112a588ef92b203c066ae64fc98ef2d0` |

Paired full-context evaluation 16495 completed on four A100s in 39:34. All
120 expected `(problem, sample)` pairs were present, pairing was verified, the
official grader emitted no warnings, and no response hit the evaluation
length cutoff.

| Paired AIME24 metric | Untouched | CH1 step 50 | Delta | Paired 95% CI |
|---|---:|---:|---:|---:|
| Avg@4 | 0.775000 | 0.716667 | -0.058333 | [-0.125000, +0.008333] |
| Majority@4 | 0.800000 | 0.833333 | +0.033333 | [-0.066667, +0.133333] |
| Pass@4 | 0.866667 | 0.833333 | -0.033333 | [-0.100000, 0.000000] |

At the paired-sample level, 5 answers improved, 12 degraded, 81 stayed
correct, and 22 stayed wrong. The negative Avg@4 result is not a truncation or
verbosity artifact: mean output length fell from 14,645.7 untouched tokens to
13,078.1 CH1 tokens, while cutoff rate remained zero. Mean tokens for correct
responses fell from 12,351.1 to 10,089.8; mean tokens for wrong responses fell
from 22,549.0 to 20,636.8. The learned policy therefore lost sample-level
accuracy despite completing its answers.

Controller 16496 applied the observed-delta gate, recorded
`stop-ch-development-no-positive-confirmation`, and submitted no paper-scale
job.

## Efficiency and conditional scale plan

Four-GPU CH1 pilot steps took approximately 8:10 on average. The GPUs generated
at about 72 tokens/second each and averaged approximately 95% utilization;
microbatch 2 is not memory-safe with the observed 40+ GiB footprint on 46 GiB
cards.

The stable eight-A100 configuration reduced mean optimizer-step time to
approximately 3:00, a 2.76x throughput improvement over the four-GPU pilot
while preserving effective batch 32. Its production settings are vLLM memory
utilization 0.32 plus expandable CUDA allocator segments. The successful
confirmation main path consumed approximately 26.3 GPU-hours: 3.4 for cache
generation, 20.3 for training, and 2.6 for evaluation. Failed pre-step memory
probes are excluded from that total.

The publication-scale cache contains the first 6,912 pinned Math-CoT-20k
identities. The fixed 5% content-hash diagnostic partition leaves exactly
6,526 eligible train identities. A 200-step effective-batch-32 run therefore
uses 6,400 first-pass examples without early recycling.

All 6,912 raw problem/reference contexts were audited before generation:

| Context audit | Tokens |
|---|---:|
| Problem+reference p99 | 14,918 |
| Problem+reference maximum | 16,607 |
| Conservative complete teacher-prompt maximum | 26,946 |
| Projected prompts above 28,672 | 0 |

Had CH1 cleared its gate, the preferred full allocation was node10's eight
GPUs with data parallelism 8 and gradient accumulation 4. The prepared chain
includes checkpoints at 50/100/150/200, automatic resume from complete
checkpoints, up to three source-pinned recovery attempts, append-only
canonical logs, and combined telemetry. It was intentionally not started
after the negative confirmation.

## Promotion and final evaluation gates

1. The 12-step CH1 pilot had exactly zero paired Avg@4 delta. This admitted
   one same-method diversity/dose confirmation, not a parameter sweep.
2. The 50-step confirmation had -0.058333 paired Avg@4 delta. It therefore
   failed the observed-delta promotion gate and ended CH development.
3. The fresh 6,912-row cache and 200-step paper-scale training run were not
   launched.
4. Official Avg@12 paper-scale AIME24 and locked AIME25/AIME26 remain
   untouched.
5. Any future method family must be preregistered as a new experiment rather
   than selected post hoc from CH checkpoints.

## Current conclusion

G4 is slightly positive but uncertain and trained on only 22 identities. CH0
is slightly negative and severely rollout-truncated. CH1 fixes the measured
completion failure while retaining every valid example and preserving
free-form dossiers, but increasing exposure from 12 to 50 steps changed its
paired Avg@4 effect from 0.000000 to -0.058333. No tested method supports a
publication-scale positive claim.

The bounded autoresearch phase is therefore a valid negative result. It shows
that longer blind attempts improve dossier coverage, and that eight-A100
training can be made stable and 2.76x faster, but neither improvement implies
better downstream reasoning. Further CH checkpoint selection would violate
the bounded gate and risk development-set overfitting. The next scientifically
distinct study, if separately authorized, should retain the dynamic natural
language dossier and complete coverage while testing an objective with direct
outcome grounding (for example an OPSD-plus-GRPO control), beginning with a
small preregistered pilot rather than the prepared paper-scale CH run.
