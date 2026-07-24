# End-to-end implementation and run order

## Locked scope

The two untouched checkpoints are `Qwen/Qwen3-1.7B` and
`Qwen/Qwen3-4B`, each at the immutable revision in its YAML file. These are
post-trained instruction checkpoints, never `*-Base`. Every tokenizer call
uses `enable_thinking=True`; the evaluator records whether a non-empty
thinking segment was actually produced.

The official OPSD trainer, MathArena datasets, and LiveCodeBench schema,
extractor, and execution checker are Git submodules. Configs pin every model
and dataset revision. The local OPSD wrapper changes only the official
launcher's hard-coded dataset: it supplies pinned Math-CoT-20k and renames
`question,response` to the official trainer's required `problem,solution`
schema.

## Turing layout

Small source files and Slurm logs live at
`~/OPSD-thinking-research`. Environments, Hugging Face caches, raw
generations, telemetry, WandB offline logs, and checkpoints live at
`/scratch/node10/$USER/opsd-thinking-research`. Each job refuses to start
with less than 200 GiB free. No unrelated home or scratch data is deleted.

## Run order

1. Sync the source tree and create `logs/`.
2. Submit `infra/turing/setup_env.sbatch`. It installs the official OPSD
   dependency versions, caches both models and all four datasets, validates
   row counts, checks CUDA/FlashAttention/vLLM, and asserts that the Qwen
   thinking-mode template switch changes the rendered prompt correctly.
3. Run untouched 1.7B math configs, then untouched 1.7B LCB.
4. Run untouched 4B math configs, then untouched 4B LCB.
5. Score each LCB generation job separately with the official execution
   checker.
6. Train 1.7B OPSD for exactly 200 optimizer steps and inspect its saved
   `generations/` files before evaluation.
7. Evaluate steps 50/100/150/200 on AIME 2025 and HMMT 2025. Evaluate only
   step 200 on AIME 2026 and LCB v6.
8. Repeat steps 6–7 for 4B.
9. Copy summaries, review packets, final adapters, manifests, and telemetry
   back to `reproductions/05_results/`.

The generic evaluation launcher uses eight one-GPU workers. Each worker owns
an independent JSONL shard with deterministic seeds, so preemption can resume
without duplicates. LoRA checkpoints are loaded directly by vLLM.

## Observability and acceptance

`infra/turing/monitor.sh` shows queue state, accounting state, exit codes,
and recent log tails. Every GPU job records 30-second utilization, memory,
power, and temperature telemetry. Generation records contain the prompt
hash, seed, full response, separated reasoning/final text, token count,
finish reason, and task-specific extraction fields.

A run is accepted only if:

- its Slurm state is `COMPLETED` with exit code `0:0`;
- every expected `(problem_id, sample_index)` pair exists exactly once;
- every record says thinking was enabled and the aggregate reports the
  non-empty-thinking rate;
- cutoff, extraction/formatting, and output-length diagnostics are reported;
- LCB uses the official execution checker;
- sample correct and incorrect rollouts have been manually read;
- the immutable config, package freeze, upstream SHAs, and adapter are
  retained.

The primary paper comparison is untouched versus step 200. Intermediate
checkpoint curves are diagnostic and use only AIME 2025 and HMMT 2025, so
AIME 2026 and LCB v6 are not used to select a checkpoint.
