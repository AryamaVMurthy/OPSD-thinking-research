# Qwen3-4B thinking-enabled OPSD

This is the independent eight-GPU counterpart of the 1.7B reproduction.
The effective batch is 32 (one example per GPU, four accumulation steps), with
checkpoints at steps 50, 100, 150, and 200.

The beta=0 full-vocabulary forward KL is reduced in 4,096-token vocabulary
chunks to bound temporary memory. This is mathematically the same loss, not
the upstream top-k approximation.
