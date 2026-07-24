# Thinking-Enabled OPSD Reproduction

This repository reproduces the untouched, post-trained Qwen3 instruction
checkpoints and a thinking-enabled OPSD-Standard baseline on Turing.

Only these model IDs are valid:

- `Qwen/Qwen3-1.7B`
- `Qwen/Qwen3-4B`

The pretrained `*-Base` checkpoints are rejected by configuration validation.
Thinking mode is required for student rollouts, privileged-teacher passes, and
all benchmark generation.

## Reproductions

Each experiment is deliberately independent:

1. [`reproductions/01_qwen3_thinking_math`](reproductions/01_qwen3_thinking_math)
2. [`reproductions/02_qwen3_thinking_livecodebench`](reproductions/02_qwen3_thinking_livecodebench)
3. [`reproductions/03_opsd_thinking_1p7b`](reproductions/03_opsd_thinking_1p7b)
4. [`reproductions/04_opsd_thinking_4b`](reproductions/04_opsd_thinking_4b)
5. [`reproductions/05_results`](reproductions/05_results)

The implementation and cluster runbook are in [`docs/implementation_guide.md`](docs/implementation_guide.md).

## Quick validation

```bash
python3 -m unittest discover -s tests -v
python3 -m opsd_research.cli validate-all
```

## Upstream sources

Pinned upstream repositories live under `third_party/`:

- OPSD: `7448751f307a9cdbcc1246dd1565a1a605b443df`
- MathArena: `a11194deff8c67a232974a383795e8a2776b4c6f`
- LiveCodeBench: `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24`
