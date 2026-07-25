# End-to-end reproduction status

Snapshot: 2026-07-25 12:40 IST

Queue update at 2026-07-25 12:48 IST: LiveCodeBench was explicitly
deferred. Jobs 16078, 16079, 16089, and 16090 were cancelled, with the partial
1.7B generations retained on scratch. Job 16080, the Qwen3-4B OPSD step-50
AIME 2025 evaluation, started immediately on all eight A100s. The active
priority is now the complete nine-run 4B math series.

This report separates accepted results from work that is still running or
queued. A result is accepted only after Slurm success, exact generation-matrix
validation, checksum verification, official scoring, comparison with the
matching untouched run, manual rollout review, and immutable archiving.

## Executive status

| Phase | Status | Completed |
|---|---|---:|
| Turing infrastructure and validation gates | Ready | all required paths |
| Untouched Qwen3 thinking baselines | Complete | 8/8 benchmark/model runs |
| OPSD-Standard training | Complete | 2/2 models |
| OPSD post-training evaluations | In progress | 7/20 accepted |
| Current live work | Running | 1.7B step-200 LiveCodeBench generation |
| Remaining post-training work | Queued | 2 missing 1.7B math runs and all 10 4B runs |
| Final cross-model analysis | Waiting | begins after all 20 evaluations are accepted |

The untouched baselines and both 200-step OPSD training runs are complete and
archived. Seven 1.7B math diagnostics are accepted. Every accepted 1.7B
post-training result is below its paired untouched baseline. This is a real
negative reproduction result, not a scorer or formatting artifact.

## Exact experimental scope

The current reproduction phase covers:

- instruction-tuned `Qwen/Qwen3-1.7B` and `Qwen/Qwen3-4B`, never the Base
  variants;
- thinking enabled for untouched inference and for both the OPSD student and
  privileged teacher;
- AIME 2025, HMMT February 2025, AIME 2026, and LiveCodeBench v6;
- untouched inference followed by OPSD-Standard training and checkpoint
  evaluation at steps 50, 100, 150, and 200;
- AIME 2026 and LiveCodeBench only at step 200, as held-out/final diagnostics.

Because OPSD is trained only on Math-CoT-20k, LiveCodeBench is an
out-of-domain transfer/regression diagnostic. It is not evidence of
code-domain training or a targeted claim that this adapter improves coding.
Targeted coding OPSD would require a separate leakage-controlled code-training
corpus and a separately trained adapter.

BFCL, SearchQA, GRPO, Purified OPSD, J-space, and String Seed of Thought are
not part of the running phase. They remain possible later experiments and
must not be described as completed.

Pinned inputs:

| Item | Immutable revision |
|---|---|
| Qwen3-1.7B | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| Qwen3-4B | `1cfa9a7208912126459214e8b04321603b3df60c` |
| Math-CoT-20k training set | `1435fb21d4fecc8ad4966a26f22a874cf2b527f1` |
| AIME 2025 data | `c94da77eb22bbd6439e62a323bec18493a421302` |
| HMMT 2025 data | `6fdc4277120810ff75aa22d2d5489b91f7a262a1` |
| AIME 2026 data | `d2de22f3c656b4f56cf8981212186377d1e23bc3` |
| LiveCodeBench data | `0fe84c3912ea0c4d4a78037083943e8f0c4dd505` |
| MathArena scorer | `a11194deff8c67a232974a383795e8a2776b4c6f` |

The source checkout used by the active Turing queue is deliberately frozen at
`aee3d8dac119ae0815aa6c8074ad7af76361d2ee`. It will not be changed until
the queue drains, because changing it during a job previously caused a stale
NFS handle. The current local/GitHub branch head is `65c9fc6`.

## Hardware and cluster allocation

All GPU jobs run on Turing `node10`, Slurm partition `u22`, QOS `high`, under
account `priyesh.shukla`.

| Property | Value |
|---|---|
| GPU count | 8 |
| GPU model | NVIDIA A100-SXM4-40GB |
| Memory per GPU | 40,960 MiB |
| Total physical VRAM | 320 GB |
| NVIDIA driver | 570.211.01 |
| CUDA module | 12.4 |
| Python | 3.10.20 |
| Training CPUs / RAM, 1.7B | 32 CPUs / 384 GB |
| Training CPUs / RAM, 4B | 64 CPUs / 768 GB |
| Generation CPUs / RAM | 8 tasks × 8 CPUs / 512 GB |
| LCB scoring | 32 CPU processes / 128 GB |

The LiveCodeBench scorer itself executes candidate programs on CPUs. Turing
accounting nevertheless records a two-GPU allocation for the scoring jobs
because of the site's default TRES policy; those GPUs are not used by the
scoring code.

## Exact software stack

The active scratch environment contains:

```text
accelerate==1.11.0
datasets==3.6.0
deepspeed==0.18.2
flash-attn==2.8.3
numpy==2.2.6
peft==0.17.1
safetensors==0.5.3
tokenizers==0.22.2
torch==2.8.0
transformers==4.57.1
trl==0.26.0
vllm==0.11.0
```

## Parallelism and memory strategy

### Training

Training uses one node and eight processes through Hugging Face Accelerate.
DeepSpeed ZeRO stage 2 partitions optimizer states and gradients across the
eight ranks. Optimizer state is offloaded to CPU, communication overlap and
contiguous gradients are enabled, and computation uses BF16.

Additional memory and throughput choices are:

- LoRA rather than full-parameter updates;
- LoRA rank 64, alpha 128, targeting `q_proj`, `k_proj`, `v_proj`, `o_proj`,
  `gate_proj`, `up_proj`, and `down_proj`;
- gradient checkpointing and FlashAttention 2;
- per-device microbatch 1 and gradient accumulation 4;
- effective optimizer batch `8 GPUs × 1 × 4 = 32`;
- colocated vLLM rollout generation with tensor parallel size 1 and
  `gpu_memory_utilization=0.6`;
- no pipeline parallelism, no tensor parallelism across GPUs, no FSDP, and no
  multi-node execution.

For 4B only, the implementation computes logits only for the generated tail
and chunks the exact full-vocabulary objective in blocks of 4,096 vocabulary
entries. A pinned-Transformers test verified identical selected logits and
LM-head gradients against the unoptimized calculation. This is a
memory-allocation repair, not an approximation of the OPSD loss.

### Evaluation

Evaluation uses eight independent Slurm tasks, each pinned to one A100. The
dataset is sharded across the eight tasks, so this is eight-way data
parallelism. Each task runs one vLLM engine with tensor parallel size 1.
Workers use deterministic sample-specific seeds and disjoint vLLM port
blocks. vLLM uses BF16, `gpu_memory_utilization=0.90`,
`max_model_len=40960`, and eager execution. Post-training runs load the LoRA
adapter directly into vLLM.

This arrangement avoids synchronization between inference workers and has
been faster than placing one model across all GPUs for these 1.7B and 4B
models.

## Protocol parameters

### Untouched and post-training math inference

| Parameter | Value |
|---|---:|
| Samples per problem | 12 |
| Temperature | 1.0 |
| Top-p | 0.95 |
| Top-k | disabled (`-1`) |
| Maximum new tokens | 38,912 |
| Model context | 40,960 |
| Base seed | 42 |
| Metrics | Avg@12, Maj@12, Pass@12 |

### Untouched and post-training LiveCodeBench v6

| Parameter | Value |
|---|---:|
| Problems | 175 |
| Samples per problem | 10 |
| Total generations per model/checkpoint | 1,750 |
| Temperature | 0.6 |
| Top-p | 0.95 |
| Top-k | 20 |
| Maximum new tokens | 38,912 |
| Model context | 40,960 |
| Base seed | 42 |
| Metrics | official execution Pass@1 and Pass@5 |

### OPSD-Standard training

| Parameter | Value |
|---|---:|
| Fixed privileged teacher | yes |
| Student/teacher thinking | enabled/enabled |
| Optimizer steps | 200 |
| Checkpoints | 50, 100, 150, 200 |
| Learning rate | `5e-6` |
| Maximum gradient norm | 0.1 |
| Completion samples per prompt | 1 |
| Completion temperature | 1.1 |
| Top-p / top-k | 0.95 / 20 |
| Maximum completion length | 1,024 |
| Generalized JSD beta | 0.0 |
| Objective at beta 0 | forward KL, teacher to student |
| Per-vocabulary-element clip | 0.05 |
| LoRA | rank 64, alpha 128 |
| Seed / data seed | 42 / 42 |

The 1,024-token completion budget matches the selected OPSD reproduction
protocol. With thinking enabled, however, it truncates nearly every training
trajectory. The result is therefore faithful to this protocol but mostly
distills prefixes rather than completed reasoning-answer trajectories.

## Completed untouched baselines

All eight untouched runs completed with exit code `0:0`. Math was rescored
with the pinned official MathArena parser/equivalence checker. LiveCodeBench
used the official execution checker.

### Math

| Model | Benchmark | Avg@12 | Maj@12 | Pass@12 | Format | Length cutoff |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-1.7B | AIME 2025 | 0.3611 | 0.4333 | 0.6667 | 0.9917 | 0.0167 |
| Qwen3-1.7B | HMMT Feb 2025 | 0.2278 | 0.2333 | 0.5333 | 1.0000 | 0.0028 |
| Qwen3-1.7B | AIME 2026 | 0.3833 | 0.5000 | 0.6333 | 0.9917 | 0.0111 |
| Qwen3-4B | AIME 2025 | 0.6639 | 0.8000 | 0.9000 | 0.9583 | 0.0500 |
| Qwen3-4B | HMMT Feb 2025 | 0.4417 | 0.5333 | 0.6667 | 0.9861 | 0.0222 |
| Qwen3-4B | AIME 2026 | 0.6278 | 0.7333 | 0.8000 | 0.9500 | 0.0528 |

### LiveCodeBench v6

| Model | Pass@1 | Pass@5 | Code extraction | Length cutoff | Mean output tokens |
|---|---:|---:|---:|---:|---:|
| Qwen3-1.7B | 0.2903 | 0.3618 | 0.9726 | 0.0280 | 15,041 |
| Qwen3-4B | 0.4783 | 0.5715 | 0.9520 | 0.0480 | 14,705 |

Both LiveCodeBench archives contain exactly 1,750 unique generation records
and 1,750 graded records covering all 175 problems. Thinking was enabled in
every record. The archive checksums and manual failure examples are recorded
in [`untouched_baselines.md`](untouched_baselines.md).

## Completed OPSD training

| Statistic | Qwen3-1.7B | Qwen3-4B |
|---|---:|---:|
| Slurm job | 16051 | 16057 |
| Wall time | 1:30:40 | 2:32:12 |
| Final step | 200 | 200 |
| Final loss | -0.0173 | -0.0174 |
| Final gradient norm | 0.0221662 | 0.0286546 |
| Logged vLLM calls | 6,366 | 6,338 |
| Calls exactly at 1,024 tokens | 6,350 (99.75%) | 6,335 (99.95%) |
| Mean completion tokens | 1,023.15 | 1,023.856 |
| Saved rank-0 rollouts | 784 | 800 |
| Rollouts closing `</think>` | 17 (2.17%) | 1 (0.125%) |
| Rollouts with boxed answer | 16 (2.04%) | 0 |
| Peak GPU memory | 37,211 MiB | 38,887 MiB |
| Mean GPU utilization | 85–87% | 90.5–91.4% |
| Maximum GPU temperature | 62°C | 67°C |
| Minimal evidence archive | 418 MiB | 788 MiB |

The negative logged loss is explainable from the upstream objective:
individual vocabulary KL summands are clipped before summation. Some
individual summands are negative, so clipping away positive mass destroys the
usual non-negative-divergence guarantee. Gradients remained finite and both
runs completed normally.

Checkpoint-200 adapter SHA-256:

- 1.7B:
  `18b17aedf89519813f5f9e352412d503f03a8c5c73519546dd39f252c0053042`;
- 4B:
  `68cbcedfe4331707ed8974dfaf52e10613cc4ad0a6245ab68c655cf979983c62`.

Evidence archive SHA-256:

- 1.7B:
  `7269e5845358e1b35c130ffc0bdc28c33941d9d434749b2cb036a3d7a67f97f5`;
- 4B:
  `78b1d9b8129cf16edb12478ee2f45522a4f4de1c497bba592b11f496e493c348`.

Full resumable DeepSpeed checkpoints remain on node10 scratch. Minimal
adapters, trainer state, generation dumps, logs, telemetry, manifests, and
manual review packets have been copied locally and verified.

## Accepted 1.7B post-training results

| Step | Benchmark | Avg@12 | Maj@12 | Pass@12 | Paired Avg delta | Delta 95% CI |
|---:|---|---:|---:|---:|---:|---:|
| 50 | AIME 2025 | 0.2778 | 0.4000 | 0.6000 | -0.0833 | [-0.1417, -0.0306] |
| 100 | HMMT Feb 2025 | 0.1417 | 0.2000 | 0.3333 | -0.0861 | [-0.1361, -0.0389] |
| 150 | AIME 2025 | 0.1944 | 0.2667 | 0.4333 | -0.1667 | [-0.2583, -0.0889] |
| 150 | HMMT Feb 2025 | 0.1167 | 0.2000 | 0.2333 | -0.1111 | [-0.1889, -0.0389] |
| 200 | AIME 2025 | 0.1611 | 0.1667 | 0.3333 | -0.2000 | [-0.2889, -0.1222] |
| 200 | HMMT Feb 2025 | 0.0833 | 0.1667 | 0.2667 | -0.1444 | [-0.2250, -0.0750] |
| 200 | AIME 2026 | 0.1472 | 0.2333 | 0.3667 | -0.2361 | [-0.3361, -0.1389] |

The confidence intervals use 10,000 paired problem-cluster bootstrap samples.
Prompt hashes, problem identities, sample indices, and seeds match the
untouched runs. Manual inspection finds genuine reasoning errors, unsupported
assumptions, repetition, and missing thinking closures. The short training
rollout budget is a strong mechanism to test, but it is not yet proven to be
the sole cause: data quality and the clipped objective are independent
concerns.

## Runtime statistics

| Work | Job | Wall time |
|---|---:|---:|
| 1.7B AIME 2025 baseline | 16017 | 0:23:38 |
| 1.7B HMMT 2025 baseline | 16018 | 0:23:43 |
| 1.7B AIME 2026 baseline | 16019 | 0:22:54 |
| 4B AIME 2025 baseline | 16020 | 0:37:35 |
| 4B HMMT 2025 baseline | 16021 | 0:36:38 |
| 4B AIME 2026 baseline | 16022 | 0:31:33 |
| 1.7B LiveCodeBench generation baseline | 16023 | 1:23:36 |
| 4B LiveCodeBench generation baseline | 16024 | 2:09:53 |
| 1.7B / 4B LiveCodeBench scoring | 16025 / 16026 | 0:01:51 / 0:02:04 |
| 1.7B OPSD training | 16051 | 1:30:40 |
| 4B OPSD training | 16057 | 2:32:12 |
| Accepted 1.7B post-train math jobs | 16069, 16072–16077 | 0:17:59–0:33:13 each |

## Live queue at this snapshot

Job 16078, the 1.7B step-200 LiveCodeBench generation, is running. At 1:21:37
elapsed, two of eight shards had completed and six workers remained active.
The two persisted shards contain 437 records; the final accepted matrix must
contain 1,750. There were no logged worker errors. Its official CPU scoring
job 16079 is dependency-queued and starts only if generation exits
successfully.

Preliminary shard statistics are operational diagnostics only, not a reported
benchmark result. Completed shard 0 has 219 records, 12 length cutoffs
(5.48%), 211 closed thinking segments, and 207 extractable code responses.

The remaining queue is:

| Job | Planned run | State at snapshot |
|---:|---|---|
| 16097 | 1.7B step-50 HMMT retry | pending |
| 16098 | 1.7B step-100 AIME retry | pending |
| 16080–16087 | 4B steps 50/100/150/200, AIME and HMMT | pending |
| 16088 | 4B step-200 AIME 2026 | pending |
| 16089 | 4B step-200 LiveCodeBench generation | pending |
| 16090 | 4B LiveCodeBench official scoring | dependency-pending |

The GPU evaluations intentionally run sequentially because each uses all
eight available A100s. Slurm reports `AssocGrpCpuLimit` for the waiting GPU
jobs. This is an account scheduling constraint, not a failed dependency or
code problem.

Based on measured run times, approximately 9–11 hours of aggregate queue
execution remained at the snapshot, excluding additional cluster wait time.

## Failures found and fixed

Two incomplete results are preserved rather than silently discarded:

- job 16070 failed before generation because eight vLLM workers selected the
  same port; workers now receive deterministic disjoint 16-port blocks and
  retry 16097 is queued;
- job 16071 stopped after 45 valid samples due to a stale NFS handle after
  the live source checkout changed; the remote source is now frozen and retry
  16098 is queued.

The first 4B smoke attempt also exposed avoidable full-prompt logits and an
out-of-memory failure. The exact tail-logits and vocabulary-chunk repair was
proved against the original logits and gradients before the full 4B run was
allowed to proceed.

An upstream final-buffer condition omitted the last rollout dump after a
completed training step. The wrapper now flushes the non-empty buffer after
`train()` and records a structured event. This changes logging only.

## Validation, logging, and storage

Implemented safeguards include:

- immutable model, dataset, scorer, adapter, and source revisions;
- config validation that rejects Base models, thinking-off runs, unpinned
  revisions, inconsistent effective batches, and unsafe 4B settings;
- exact `(problem_id, sample_index)` coverage and uniqueness checks;
- per-file SHA-256 transfer and artifact manifests;
- official MathArena rescoring sidecars without mutating raw generations;
- official LiveCodeBench execution scoring;
- paired treatment/baseline seed and prompt-hash verification;
- 10,000-sample paired bootstrap comparisons;
- deterministic manual-review packets containing correct, incorrect, cutoff,
  and changed-outcome examples;
- Slurm logs, adapter metadata, training summaries, W&B offline logs, and
  per-GPU `nvidia-smi` telemetry;
- resumable DeepSpeed checkpoints and separate minimal evidence archives.

The current automated suite passes:

```text
49 passed, 5 skipped, 10 subtests passed
```

The five local skips require Torch and are exercised in the pinned Turing
environment; the critical 4B logits/gradient equivalence test passed there.

Storage at the snapshot:

| Location | Used | Free | Action |
|---|---:|---:|---|
| Turing home quota | 50 GB / 50 GB | about 70 MB | keep code/logs only; do not add artifacts |
| node10 scratch filesystem | 8.1 TB / 14 TB | 5.2 TB | models, environments, checkpoints, generations |
| scratch OPSD project | about 51 GB | within scratch | retain until final archive audit |

Home quota is the main operational risk. No unrelated user data has been
deleted. Large experiment data is already isolated on scratch, so the running
queue has remained stable with approximately 70 MB of home headroom.

## Work still required

To close the current reproduction phase:

1. Let job 16078 finish, run official scorer 16079, verify all 1,750 records,
   compare against untouched 1.7B, manually inspect changed and cutoff
   rollouts, and archive the result.
2. Complete retries 16097 and 16098, then officially rescore, compare, review,
   and archive them.
3. Complete the eight 4B checkpoint-curve math runs, step-200 AIME 2026, and
   step-200 LiveCodeBench plus its official scorer.
4. Apply the same acceptance gates to every result: Slurm success, complete
   unique matrix, checksums, official score, paired comparison, manual review,
   and immutable archive.
5. Assemble final checkpoint curves and untouched-versus-OPSD tables for both
   model sizes, verify every archive hash, and write the reproduction
   conclusion.
6. Only after that evidence is closed, choose the next causal experiment.

The most informative next research design is a 2×2 ablation:

- OPSD-Standard with 1,024-token training rollouts: already complete;
- OPSD-Standard with longer or adaptive rollouts;
- Purified OPSD with 1,024-token rollouts;
- Purified OPSD with longer or adaptive rollouts.

That design separates truncation from objective/data-quality effects. GRPO
should then be added as a training-control baseline before attributing gains
to a new J-space or String Seed of Thought method.
