# GRAF / dynamic contrastive-hindsight OPSD autoresearch

Status: active. This report records only completed, checksum-verified results.
CH1 pilot training and its paired evaluation are still pending.

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

## CH1 training status

CH1 cache 16444 is immutable and checksum-verified. A one-step engineering
smoke completed generation, backward, and optimization without OOM:

| Smoke metric | Value |
|---|---:|
| Loss | 0.0004 |
| Gradient norm | 0.0210 |
| Peak observed memory | ~40.7 / 46.1 GiB per GPU |
| GPU utilization | ~96–97% |

Pilot 16460 is the clean 12-step token-matched run. Its paired full-context
AIME24 evaluation is dependency-gated by controller 16461. Results will be
added only after adapter, rollout, checksum, scorer, and paired-comparison
validation.

## Efficiency and scale plan

Four-GPU CH1 pilot steps take approximately 8:27. The GPUs generate at about
72 tokens/second each and remain approximately 96% utilized; microbatch 2 is
not memory-safe with the observed 40+ GiB footprint on 46 GiB cards.

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

Preferred full allocation is node10's eight GPUs with data parallelism 8 and
gradient accumulation 4. This preserves effective batch 32 while roughly
halving step wall time versus four GPUs. The chain includes checkpoints at
50/100/150/200, automatic resume from complete checkpoints, up to three
source-pinned recovery attempts, append-only canonical logs, and combined
telemetry.

## Promotion and final evaluation gates

1. CH1 pilot must have positive paired full-context AIME24 Avg@4 delta versus
   untouched. Otherwise no paper-scale CH1 job is launched.
2. A positive pilot launches a fresh 6,912-row cache and 200-step training run.
3. Full CH1 and untouched models are evaluated sequentially on official
   12-sample AIME24.
4. Locked AIME25/AIME26 remain untouched unless full AIME24 delta is positive.
5. A positive paper claim requires the paired confidence-interval lower bound
   to exceed zero on both locked benchmarks. Otherwise the method is reported
   as a development result.

## Current conclusion

G4 is slightly positive but uncertain and trained on too few identities. CH0
is slightly negative and severely truncated. CH1 fixes the principal measured
failure mode while preserving 100% cache coverage and free-form reasoning.
Its paired result is still required before any claim or full-scale launch.
