# Untouched Qwen3 thinking-mode LiveCodeBench

This uses the official pinned LiveCodeBench data model, extraction routine,
and execution checker. It evaluates only the 175-problem v6 slice with ten
samples per problem. The label is **LiveCodeBench v6 — Qwen3 thinking
profile**, because its temperature 0.6, top-p 0.95, top-k 20, and thinking
mode are Qwen3 recommendations rather than the standard LCB leaderboard
profile.

Generation and execution scoring are separate Slurm jobs so untrusted
generated code never runs inside an inference worker.
