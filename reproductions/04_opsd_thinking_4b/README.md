# Qwen3-4B thinking-enabled OPSD

This is the independent eight-GPU counterpart of the 1.7B reproduction.
The effective batch is 32 (one example per GPU, four accumulation steps), with
checkpoints at steps 50, 100, 150, and 200.

The beta=0 full-vocabulary forward KL is reduced in 4,096-token vocabulary
chunks to bound temporary memory. This is mathematically the same loss, not
the upstream top-k approximation.

Before the 200-step run, submit `smoke.sbatch`. It performs one complete
optimizer step with the final eight-GPU microbatch/accumulation layout, saves
a checkpoint, and emits the same training-health summary. The main run is
submitted only if this job completes without CUDA errors and its adapter,
trainer state, rollout dump, and telemetry are present.
