# OPSD post-training evaluation progress

This file records only retrieved, checksum-verified, officially scored, and
manually reviewed results. Runs still in Slurm or awaiting retrieval are not
reported as completed.

Math correctness uses pinned MathArena commit
`a11194deff8c67a232974a383795e8a2776b4c6f`. Deltas and confidence intervals
use paired problem-cluster bootstrap resampling against the matching untouched
model, with identical problem/sample seeds and prompt hashes verified first.

## Qwen3-1.7B completed diagnostics

| Step | Benchmark | Avg@12 | Maj@12 | Pass@12 | Paired Avg delta | Delta 95% CI |
|---:|---|---:|---:|---:|---:|---:|
| 50 | AIME 2025 | 0.2778 | 0.4000 | 0.6000 | -0.0833 | [-0.1417, -0.0306] |
| 100 | HMMT Feb 2025 | 0.1417 | 0.2000 | 0.3333 | -0.0861 | [-0.1361, -0.0389] |
| 150 | AIME 2025 | 0.1944 | 0.2667 | 0.4333 | -0.1667 | [-0.2583, -0.0889] |

These three intermediate results all trail the untouched model on Avg@12.
Manual review attributes the gap to genuine reasoning mistakes, unsupported
assumptions, repetitive long-tail failures, and—in the step-150 AIME
run—missing `</think>` closures. The official 1,024-token training rollout cap
truncates almost all thinking rollouts and is the leading mechanistic
hypothesis for this degradation.

## Verified archives

```text
22a219e097fc4ba451c4271b3fd587839dbf4dc846de69049f79d1291dafa10d  opsd-qwen3-1p7b-step50-aime25.tar.gz
2706fad7dfe13b13d389ceea7c416b5c2d5c15e8d6b7d8469de5ef05a1351618  opsd-qwen3-1p7b-step100-hmmt25.tar.gz
914f1f8159e8d84f938cf48f4e2437d67656dc0df032ae81775ed4100c6cf5c3  opsd-qwen3-1p7b-step150-aime25.tar.gz
```

## Preserved recoveries

- Job `16070` failed before generation because eight vLLM workers raced for
  the same dynamically selected port. Its empty result/log bundle is
  preserved; deterministic disjoint shard ports fix the root cause. Retry
  job: `16097`.
- Job `16071` was interrupted after 45 valid samples by a stale NFS file
  handle caused by changing the source checkout during execution. Its partial
  bundle is preserved. The Turing source is now frozen for the entire queue.
  Retry job: `16098`.
