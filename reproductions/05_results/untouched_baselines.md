# Untouched Qwen3 thinking baselines

These are the accepted no-training baselines. All Slurm jobs completed with
exit code `0:0`; generation matrices were checked for exact
`(problem_id, sample_index)` coverage and uniqueness. Math uses 12 samples per
problem. LiveCodeBench v6 uses 10 samples per problem and the official
execution checker.

Math values below use the parser and equivalence checker from the pinned
official MathArena commit
`a11194deff8c67a232974a383795e8a2776b4c6f`. The immutable generation
records retain their original `correct` field, while an SHA-256-linked
`official-grades.jsonl` sidecar and `summary.official.json` provide the
authoritative scores. This preserves the full audit trail.

## Math

| Model | Benchmark | Avg@12 | Maj@12 | Pass@12 | Format | Length cutoff |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B | AIME 2025 | 0.3611 | 0.4333 | 0.6667 | 0.9917 | 0.0167 |
| Qwen3-1.7B | HMMT Feb 2025 | 0.2278 | 0.2333 | 0.5333 | 1.0000 | 0.0028 |
| Qwen3-1.7B | AIME 2026 | 0.3833 | 0.5000 | 0.6333 | 0.9917 | 0.0111 |
| Qwen3-4B | AIME 2025 | 0.6639 | 0.8000 | 0.9000 | 0.9583 | 0.0500 |
| Qwen3-4B | HMMT Feb 2025 | 0.4417 | 0.5333 | 0.6667 | 0.9861 | 0.0222 |
| Qwen3-4B | AIME 2026 | 0.6278 | 0.7333 | 0.8000 | 0.9500 | 0.0528 |

Generation jobs: 16017–16022.

## LiveCodeBench v6

| Model | Pass@1 | Pass@5 | Code extraction | Length cutoff | Mean output tokens |
|---|---:|---:|---:|---:|---:|
| Qwen3-1.7B | 0.2903 | 0.3618 | 0.9726 | 0.0280 | 15,041 |
| Qwen3-4B | 0.4783 | 0.5715 | 0.9520 | 0.0480 | 14,705 |

Generation jobs: 16023 and 16024. Official scoring jobs: 16025 and 16026.

Manual checks found:

- A short `abc388_a` solution from each model was correct and used a sound
  direct construction.
- A 1.7B `abc394_e` failure attempted an exponential whole-string palindrome
  search and timed out.
- A 4B `problem3697` failure used a per-target greedy assignment that missed
  sharing through a common LCM; the checker produced a counterexample with
  model output 2 and expected output 1.
- Length-cutoff samples from both models often ended before executable code
  was emitted, explaining part of the extraction gap.

## Immutable archive checksums

```text
b755f96fd02b2ba11e90a6cbbfd76b4d859359e9f9d388647a5e5934fcc5fbff  untouched-qwen3-1p7b-aime25.tar.gz
30e79b6451f58ba9f82785a8ea6ae3d26ffdc5b8a88b7f6fb49e886e4eec0db0  untouched-qwen3-1p7b-aime26.tar.gz
5b9675fe5adbcb7cbffe0f1a602e8e3b1e871a8b80d7dba8daee08eb634fc98f  untouched-qwen3-1p7b-hmmt25.tar.gz
d00bc5cb609b66cc30dd2a377b75dcca965a99058835d1585d60cb564442ed17  untouched-qwen3-1p7b-lcb-v6.tar.gz
b4a825640c4a2c6c10c5ada985502182cec85dc2074cf0609f6bdeb2979112b3  untouched-qwen3-4b-aime25.tar.gz
80176831d392dab28742ba49fe72b24eb6b3514e7b52b782b522a53cd6d36547  untouched-qwen3-4b-aime26.tar.gz
98b0e049068f1e186e428aa4703919113fbb9932da0987981e22531ac05d5506  untouched-qwen3-4b-hmmt25.tar.gz
115974ff285d7442812ec84649f36f68fb5e08878973d981fed079f627f2fe56  untouched-qwen3-4b-lcb-v6.tar.gz
```
