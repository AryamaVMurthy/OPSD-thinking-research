# Qwen3-4B thinking-enabled OPSD

This is the independent eight-GPU counterpart of the 1.7B reproduction.
The effective batch is 32 (4 examples per GPU, one accumulation step), with
checkpoints at steps 50, 100, 150, and 200.
