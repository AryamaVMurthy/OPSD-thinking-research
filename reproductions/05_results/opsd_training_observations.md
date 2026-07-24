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
