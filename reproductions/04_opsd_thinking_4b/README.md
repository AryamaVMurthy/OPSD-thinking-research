# Qwen3-4B thinking-enabled OPSD

This is the independent eight-GPU counterpart of the 1.7B reproduction.
The effective batch is 32 (one example per GPU, four accumulation steps), with
checkpoints at steps 50, 100, 150, and 200.

The beta=0 full-vocabulary forward KL is reduced in 4,096-token vocabulary
chunks to bound temporary memory. This is mathematically the same loss, not
the upstream top-k approximation. Qwen3 is also asked to return logits only
for the final `generation_length + 1` hidden states. The upstream trainer
otherwise materializes vocabulary logits for every privileged-prompt token
and discards them immediately; tail-only logits preserve the exact scored
generation positions while avoiding that allocation.

Before the 200-step run, submit `smoke.sbatch`. It performs five complete
optimizer steps with the final eight-GPU microbatch/accumulation layout. Five
steps are required to exercise the upstream rollout-dump path as well as
checkpoint saving. The main run is submitted only if this job completes
without CUDA errors and its adapter, trainer state, rollout dump, health
summary, and telemetry are present.
