# Untouched Qwen3 thinking-mode math baselines

This folder evaluates the immutable Qwen3-1.7B and Qwen3-4B instruction
checkpoints on AIME 2025, AIME 2026, and HMMT February 2025.

The locked protocol is 12 independently seeded samples per problem,
temperature 1.0, top-p 0.95, unrestricted top-k, and 38,912 maximum new
tokens. Thinking mode is asserted in the rendered Qwen chat template. Avg@12
is primary; Pass@12 and Maj@12 are secondary.

Submit one configuration at a time:

```bash
infra/turing/submit_eval.sh \
  reproductions/01_qwen3_thinking_math/configs/qwen3-1p7b-aime25.yaml \
  math_eval qwen3-1p7b-untouched-aime25
```
