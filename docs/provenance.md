# Protocol provenance

| Component | Pinned source | Use |
|---|---|---|
| OPSD | `7448751f307a9cdbcc1246dd1565a1a605b443df` | official trainer, data collator, ZeRO-2 config, training hyperparameters |
| MathArena | `a11194deff8c67a232974a383795e8a2776b4c6f` | official AIME/HMMT datasets |
| LiveCodeBench | `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24` | official v6 schema, code extraction, and execution metrics |

The OPSD launch settings follow the upstream 1.7B and 4B scripts: BF16,
FlashAttention 2, ZeRO-2, LoRA rank 64/alpha 128 over all projection
modules, LR 5e-6, max gradient norm 0.1, rollout temperature 1.1, top-p
0.95, top-k 20, lambda 1, beta 0, privileged fixed teacher, token loss clip
0.05, and completion length 1024. This project explicitly adds
`max_steps=200`, checkpoints every 50 steps, pinned model/dataset revisions,
and thinking mode for both student and teacher.

The math decoding matches the behavior of the official OPSD evaluator:
thinking enabled, temperature 1.0, top-p 0.95 (its thinking-mode automatic
value), top-k disabled, min-p 0, presence penalty 0, 38,912 new-token cap,
and 12 samples. LCB is separately labeled as a Qwen3 thinking profile.
