# Thinking-enabled OPSD training observations

This file records implementation-level findings from the faithful 1.7B
OPSD-Standard run (Slurm job 16051, source
`e2ab5cbaf4ca8ea58d3504e18d9efe30d36dbc18`). Final benchmark results will be
added only after the 200-step run and post-training evaluations complete.

## Rollout-budget mismatch

The official OPSD protocol uses `max_completion_length=1024`. With Qwen3
thinking enabled, this is generally too short to reach a final answer:

- through optimizer step 38, 1,223 of 1,224 logged vLLM calls used exactly
  1,024 tokens;
- through the step-35 dump, only 2 of 144 saved rank-0 rollouts closed
  `</think>`;
- the step-50 dump has 20 non-empty thinking traces, 0 closed thinking traces,
  and 0 boxed answers.

The two early closed traces were manually checked. Both solved their problems
correctly, but both restarted a polished response after `</think>` and kept
generating instead of stopping. Representative open traces often begin with
sound algebra or combinatorics, then are truncated mid-derivation. Others
enter self-correction loops or develop a substantive geometric misconception
before cutoff.

This means the run is a faithful measurement of the requested protocol, but
the student is usually distilled on prefixes of reasoning rather than
complete reasoning-answer trajectories.

At step 100, the sole closed trace exposes an additional data-quality issue:
the source example asks for an English translation of an already-English
geometry sentence and requests the translated text directly, while the OPSD
collator appends a conflicting step-by-step/boxed-answer instruction. The
model repeats the sentence instead of solving it. Thus, a closed thinking tag
does not by itself imply a usable mathematical training trajectory.

## Upstream final-buffer logging gap

The upstream trainer checks its five-step rollout-save condition before the
trainer increments the final global step. The completed 1.7B run therefore
contains dumps through step 195 but no `generations_step_200.json`, even
though checkpoint 200 is complete. The wrapper now flushes any non-empty
generation buffer on the main process after `train()` returns and records a
structured flush event. This observability fix does not change optimization;
it applies to subsequent runs.

## The clipped objective is not guaranteed non-negative

For `beta=0`, the upstream loss first computes each vocabulary element of
`KL(teacher || student)`, clips each element to a maximum of 0.05, and only
then sums over vocabulary and sequence positions. Individual KL summands can
be negative even though their unmodified sum is non-negative. Capping positive
summands while retaining negative summands destroys the divergence guarantee.

The observed logged loss becomes negative and reaches `-0.0119` at checkpoint
50. Gradients remain finite, so this is not a NaN/overflow failure; it is a
consequence of the placement of the clipping operation. Any follow-up method
should distinguish this upstream “clipped JSD” objective from a true
non-negative per-sequence-token divergence.

## Execution health at checkpoint 50

- Eight A100 40 GB GPUs, microbatch 1, gradient accumulation 4, effective
  batch 32.
- Adapter rank 64, alpha 128, all seven projection module types.
- Adapter size: 139,513,368 bytes.
- Full resumable DeepSpeed checkpoint size: 1.7 GB.
- Peak observed GPU memory: 34,919 MiB.
- Mean utilization including setup: 81–84%; steady-state samples are about
  90%.
- Maximum observed temperature: 62°C.
- Checkpoint contains eight optimizer shards, model state, adapter, trainer
  state, scheduler, tokenizer, and all eight RNG states.

## Completed Qwen3-1.7B execution

Job 16051 completed all 200 optimizer steps with exit `0:0` in 1h30m40s.
The losses and gradients remained finite; the final logged loss was -0.0173,
the final gradient norm was 0.0221662, and the reported aggregate training
loss was -0.0130393.

Full resumable checkpoints 50, 100, 150, and 200 remain on node10 scratch.
The corresponding adapters, configurations, trainer states, generation
dumps, telemetry, logs, and manifests were copied locally. Every local
checkpoint checksum and the run manifest verifies. The checkpoint-200
adapter SHA-256 is
`18b17aedf89519813f5f9e352412d503f03a8c5c73519546dd39f252c0053042`.
The 418 MiB minimal evidence archive (four adapters, trainer states,
generations, log, telemetry, review packet, and manifests) has SHA-256
`7269e5845358e1b35c130ffc0bdc28c33941d9d434749b2cb036a3d7a67f97f5`.

Across the 39 saved dumps through step 195, all 784 stored rollouts started
non-empty thinking, 17 closed the thinking segment, and 16 emitted a boxed
answer. Of 6,366 logged vLLM calls, 6,350 reached exactly 1,024 tokens; mean
length was 1,023.15 and the shortest was 150. Peak memory was 37,211 MiB,
mean utilization including setup was 85–87%, and maximum temperature was
62°C. The absent step-200 dump is the upstream final-buffer bug described
above; checkpoint 200 itself is complete and verified.

## Qwen3-4B memory gate

The first 4B smoke attempt (job 16052) failed before its first optimizer step.
The upstream trainer computed vocabulary logits for every token in the long
privileged-teacher prompt and then immediately sliced those prompt positions
away. That unused tensor required another 4.10 GiB on rank 0; another rank
also had only 13.56 MiB free when an exact-KL chunk needed 20 MiB.

The repaired path asks Qwen3 for only the final
`generation_length + 1` logit positions. Dropping the final unscored position
then gives exactly the same generation-token slice as the upstream
`prompt_length - 1 : -1` expression. A pinned-Transformers Turing test
compared both prompt lengths, output logits, and LM-head gradients and passed
exactly. This changes allocation only; the full-vocabulary clipped forward-KL
objective is unchanged.

The five-step 4B gate then passed as Slurm job 16056 from source commit
`5e4af436c1b38dc72a8346863af5926f125d07af`:

- state `COMPLETED`, exit `0:0`, wall time 5m36s;
- all five logged losses finite, from -0.0029 to -0.0042;
- peak GPU memory 38,769 MiB, leaving about 2.1 GiB on a 40,960 MiB A100;
- steady training samples at 94–95% utilization, with maximum temperature
  62°C;
- full checkpoint 5 is 2,908,453,766 bytes and contains the adapter, model
  state, eight optimizer shards, eight RNG states, scheduler, tokenizer, and
  trainer state;
- the run manifest verifies, and the final-buffer repair produced
  `generations_step_5.json` with 20 non-empty samples that covers checkpoint
  5.

All 160 vLLM calls reached the 1,024-token completion cap. Manual inspection
of all 20 saved rank-0 rollouts found no closed thinking tag or boxed answer.
Several traces had a correct solution in progress before cutoff: the normal
line derivative and slope, the minimum dice score, the AM-GM minimum, the
multinomial term count, and the mark-recapture estimate. Other traces stalled
in repeated alternatives, remained at setup, depended on a missing source
figure, or had not yet resolved the main optimization. Therefore the gate
proves execution and checkpoint correctness, but it reinforces the 1.7B
finding that the official 1,024-token training budget mostly distills
incomplete reasoning prefixes when thinking is enabled.

The full 4B run provides a complementary data-quality example at step 130.
One rank-0 rollout naturally stopped after 437 tokens and closed its thinking
block, but the source record contains an already-English math problem followed
by a Chinese request to translate it to English, another English “final
version,” and the collator's conflicting request for a reasoned boxed answer.
The model coherently follows the translation instruction and repeats the
English sentence without solving the math problem. Thus, short or closed
thinking is not by itself evidence of a useful privileged trajectory; the
source task and final-answer semantics must also be validated.

## Completed Qwen3-4B execution

Job 16057 completed all 200 optimizer steps from source commit
`5e4af436c1b38dc72a8346863af5926f125d07af` with Slurm state `COMPLETED`,
exit `0:0`, and wall time 2h32m12s. The losses and gradients remained finite;
the step-200 loss was -0.0174 and its gradient norm was 0.0286546.

Full resumable checkpoints 50, 100, 150, and 200 remain on node10 scratch.
Each contains eight optimizer shards, one model-state shard, eight RNG states,
the scheduler, tokenizer, trainer state, and LoRA adapter. The checkpoint-200
directory is 2,908,479,986 bytes. All four minimal local checkpoint copies
verify against their own SHA-256 manifests. Their adapter SHA-256 values are:

- step 50:
  `4a65128c88617eb2a8bc0dfa71aa6480a84fd073fa09e260fdb416010804ed9f`;
- step 100:
  `1c867302d5574e5b707b22325c66a917ce721f897d52d5e1caafe4c8ed4307e7`;
- step 150:
  `559fb93f367dd279e9dba48f7a02035beb5c61a127ecbba697c80cb65b976a9c`;
- step 200:
  `68cbcedfe4331707ed8974dfaf52e10613cc4ad0a6245ab68c655cf979983c62`.

The final-buffer repair emitted a structured
`final_generation_buffer_flush` event and preserved
`generations_step_200.json`. Across all 40 five-step dumps, all 800 saved
rank-0 rollouts are non-empty and begin thinking; one closes `</think>` and
none emits a boxed answer. Of 6,338 logged vLLM calls, 6,335 reach exactly
1,024 tokens; the other lengths are 791, 933, and 437, giving a mean of
1,023.856 tokens. The 437-token natural completion is the malformed
translation example documented above.

Mean GPU utilization including initialization is 90.5–91.4% across the eight
A100s. Peak memory is 38,887 MiB of 40,960 MiB, and the maximum observed
temperature is 67°C. The run manifest, complete training summary, final log,
telemetry, every rollout dump, and all manual review packets are preserved
with the four minimal adapters in a 788 MiB archive. Its SHA-256 is
`78b1d9b8129cf16edb12478ee2f45522a4f4de1c497bba592b11f496e493c348`.
The gzip stream, all 12 archived checkpoint checksums, 40 generation members,
global step 200, and the 800-rollout summary were independently verified.
