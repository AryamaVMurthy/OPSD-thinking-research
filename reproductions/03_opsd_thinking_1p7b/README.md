# Qwen3-1.7B thinking-enabled OPSD

This is an independent four-GPU reproduction using the official pinned OPSD
trainer. A guard launcher redirects the trainer's hard-coded dataset to the
pinned Math-CoT-20k revision and refuses Base models, mutable revisions,
non-thinking invocations, or a run other than 200 optimizer steps.

The fixed teacher is the same frozen instruction checkpoint with LoRA
disabled. Both student rollout and privileged teacher chat templates use
thinking mode.
