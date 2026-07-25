# Official MathArena rescoring

## Why the correction was required

The original generation path used `math_verify==0.8.0`. A manual HMMT review
found that it parsed both `2^{25} \cdot 26!` and `26!` as the same value and
therefore produced false positives. It also rejected some ordinary equivalent
forms, including `\dfrac{1}{576}` versus `\frac{1}{576}`.

The model responses, seeds, prompts, and adapters are unaffected. Only the
derived math correctness labels and summaries required correction.

## Authoritative grader

- Repository: `https://github.com/eth-sri/matharena.git`
- Pinned commit: `a11194deff8c67a232974a383795e8a2776b4c6f`
- Competition configs: official AIME 2025, AIME 2026, and HMMT February 2025
  YAML files from that commit
- Parsing: `strict_parsing: false`, as specified by those configs
- Typed delimiters: enabled, matching the MathArena grader default

The rescorer calls MathArena's own `extract_answer`, `parse_answer`, and
`check_answers` functions. It writes separate sidecars:

- `official-grades.jsonl`: one compact grade record per immutable generation,
  linked to the original response by SHA-256
- `summary.official.json`: Avg@12, Maj@12, Pass@12, bootstrap interval,
  parser warnings, grade-flip counts, input hashes, config hash, and grader
  revision

The original JSONL and legacy `summary.json` are deliberately not overwritten.

## Baseline impact

| Run | Legacy Avg@12 | Official Avg@12 | Legacy Maj@12 | Official Maj@12 | Grade flips |
|---|---:|---:|---:|---:|---:|
| Qwen3-1.7B AIME 2025 | 0.3611 | 0.3611 | 0.4333 | 0.4333 | 0 |
| Qwen3-1.7B AIME 2026 | 0.3833 | 0.3833 | 0.5000 | 0.5000 | 0 |
| Qwen3-1.7B HMMT 2025 | 0.2083 | 0.2278 | 0.2333 | 0.2333 | 25 |
| Qwen3-4B AIME 2025 | 0.6611 | 0.6639 | 0.8000 | 0.8000 | 1 |
| Qwen3-4B AIME 2026 | 0.6278 | 0.6278 | 0.7333 | 0.7333 | 0 |
| Qwen3-4B HMMT 2025 | 0.3556 | 0.4417 | 0.4000 | 0.5333 | 49 |

The HMMT flips were manually spot-checked in both directions. Representative
legacy false positives answered `26!` instead of
`2^{25} \cdot 26!`. Representative legacy false negatives used equivalent
`\dfrac` fractions or radicals that MathArena correctly normalized.

## Reproduction command

```bash
python -m opsd_research.rescore_matharena \
  --config artifacts/results/<run>/config.yaml \
  --input artifacts/results/<run>/generations.shard*.jsonl \
  --grades-output artifacts/results/<run>/official-grades.jsonl \
  --summary-output artifacts/results/<run>/summary.official.json
```
