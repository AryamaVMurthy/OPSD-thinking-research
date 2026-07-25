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
| 150 | HMMT Feb 2025 | 0.1167 | 0.2000 | 0.2333 | -0.1111 | [-0.1889, -0.0389] |
| 200 | AIME 2025 | 0.1611 | 0.1667 | 0.3333 | -0.2000 | [-0.2889, -0.1222] |
| 200 | HMMT Feb 2025 | 0.0833 | 0.1667 | 0.2667 | -0.1444 | [-0.2250, -0.0750] |
| 200 | AIME 2026 | 0.1472 | 0.2333 | 0.3667 | -0.2361 | [-0.3361, -0.1389] |

These seven results all trail the untouched model on Avg@12.
Manual review attributes the gap to genuine reasoning mistakes, unsupported
assumptions, repetitive long-tail failures, and missing `</think>` closures.
The official 1,024-token training rollout cap truncates almost all thinking
rollouts and is the leading mechanistic hypothesis for this degradation.

## Qwen3-4B completed diagnostics

| Step | Benchmark | Avg@12 | Maj@12 | Pass@12 | Paired Avg delta | Delta 95% CI |
|---:|---|---:|---:|---:|---:|---:|
| 50 | AIME 2025 | 0.6472 | 0.7667 | 0.8000 | -0.0167 | [-0.0639, 0.0250] |
| 50 | HMMT Feb 2025 | 0.4361 | 0.5333 | 0.6667 | -0.0056 | [-0.0639, 0.0472] |
| 100 | AIME 2025 | 0.5639 | 0.6333 | 0.7667 | -0.1000 | [-0.1472, -0.0556] |
| 100 | HMMT Feb 2025 | 0.3694 | 0.4333 | 0.6667 | -0.0722 | [-0.1389, -0.0083] |
| 150 | AIME 2025 | 0.4417 | 0.6000 | 0.7000 | -0.2222 | [-0.3000, -0.1500] |
| 150 | HMMT Feb 2025 | 0.2917 | 0.4000 | 0.5667 | -0.1500 | [-0.2278, -0.0778] |
| 200 | AIME 2025 | 0.4139 | 0.5333 | 0.7333 | -0.2500 | [-0.3333, -0.1694] |

Step 50 is not statistically separated from the untouched Avg@12 baseline,
but Pass@12 falls from 0.9000 to 0.8000. Of 360 paired samples, 32 degrade,
26 improve, 207 remain correct, and 95 remain wrong. The run completes in
1h12m12s; mean output length is 17,646.8 tokens and 13/360 samples (3.61%)
reach the 38,912-token inference limit.

Manual paired review finds substantive changes in both directions. Degraded
examples include an incorrect coordinate/Jacobian transformation followed by
an acknowledged unsupported guess, an overconstrained combinatorial count,
and inclusion of the excluded endpoint `2pi`. Improved examples use a correct
shoelace construction for a reflected heptagon, recover the correct bounded
triangle and surface-area factor, and replace approximate rounding with an
exact coordinate construction that verifies all five distance constraints.
One both-wrong OPSD response cycles through quartic factorizations until the
length cap; a stable both-correct response retains a clean coordinate solution.

On HMMT February 2025, step 50 is also statistically indistinguishable from
the untouched Avg@12 baseline (0.4361 versus 0.4417) and retains identical
majority and pass rates. Of 360 paired samples, 33 degrade, 31 improve, 126
remain correct, and 170 remain wrong. The run completes in 1h11m35s; mean
output length is 18,447.7 tokens and 4/360 samples (1.11%) reach the
38,912-token inference limit. Manual review identifies real improvements from
complete coordinate and inclusion-exclusion arguments, alongside real
regressions caused by a missing factor for distinct rotations, a modular-sign
error, and a broken coordinate constraint. A capped geometry response shows
unproductive repetitive derivation rather than a scoring issue.

Step 100 on AIME 2025 is a statistically significant regression: Avg@12 falls
to 0.5639 from 0.6639 (paired delta -0.1000, 95% CI [-0.1472, -0.0556]);
Maj@12 falls to 0.6333 and Pass@12 to 0.7667. Of 360 paired samples, 55
degrade, 19 improve, 184 remain correct, and 102 remain wrong. The run
completed in 1h12m15s; mean output length is 18,767.8 tokens and 10/360
samples (2.78%) reach the 38,912-token cap. Manual review finds invalid
pair-level inclusion-exclusion, an unsupported chord-intersection probability,
and lost cross-row constraints; gains include a correct complement count and
an exact quartic factorization. One treatment sample hits the cap while cycling
through numerical radical guesses and emits no answer.

Step 100 on HMMT February 2025 is likewise a statistically significant
regression: Avg@12 falls to 0.3694 from 0.4417 (paired delta -0.0722, 95% CI
[-0.1389, -0.0083]); Maj@12 falls to 0.4333 while Pass@12 remains 0.6667. Of
360 paired samples, 49 degrade, 23 improve, 110 remain correct, and 178 remain
wrong. The run completed in 1h14m37s; mean output length is 19,415.0 tokens
and 10/360 samples (2.78%) reach the 38,912-token cap. Manual review finds an
unjustified substitution of a nearby rational for an irrational constant,
confusion of either-axis with both-axis separation, and asserted rather than
solved symmetric algebra. Improvements include exact coordinate and counting
arguments, but one capped geometry rollout still resolves a contradiction by
rounding to an unsupported final answer.

Step 150 on AIME 2025 is a large regression: Avg@12 falls to 0.4417 from
0.6639 (paired delta -0.2222, 95% CI [-0.3000, -0.1500]), Maj@12 falls to
0.6000, and Pass@12 to 0.7000. Of 360 paired samples, 86 degrade and only 6
improve. The run completed in 1h19m31s; mean output length is 20,291.2 tokens
and 26/360 samples (7.22%) reach the 38,912-token limit. Manual review finds
lost symbolic solutions, major combinatorial undercounts, invalid sign
repairs, and a substantially heavier long-tail failure rate. A small number of
exact geometry and perimeter solutions improve, but they do not offset the
clear degradation.

Step 150 on HMMT February 2025 is also a large regression: Avg@12 falls to
0.2917 from 0.4417 (paired delta -0.1500, 95% CI [-0.2278, -0.0778]), Maj@12
to 0.4000, and Pass@12 to 0.5667. Of 360 paired samples, 68 degrade and 14
improve. The run completed in 1h29m40s; mean output length is 20,855.9 tokens
and 23/360 samples (6.39%) reach the 38,912-token cap. Manual review finds
unsupported integer conclusions, a missing rotation factor, broken geometry
constraints, and long capped loops. Exact coordinate and modular improvements
remain, but they are too sparse to counter the broad regression.

Step 200 on AIME 2025 is the strongest AIME regression: Avg@12 falls to
0.4139 from 0.6639 (paired delta -0.2500, 95% CI [-0.3333, -0.1694]), Maj@12
to 0.5333, and Pass@12 to 0.7333. Of 360 paired samples, 104 degrade and 14
improve. The run completed in 1h22m45s; mean output length is 20,918.7 tokens
and 28/360 samples (7.78%) reach the 38,912-token cap. Manual review finds
repeated pair-versus-coordinate counting errors, incomplete branch accounting,
and capped radical-guess loops. Occasional correct combinatorial fixes do not
offset the endpoint degradation.

## Verified archives

```text
07687218dcc02994a44e42acdb8d68cbf23a12a8f8c307f8ac7d88602a6e5836  opsd-qwen3-1p7b-step50-aime25.tar.gz
e767efdf11ba08a0af60693c585936d5b2ed87ae02ccbdb54a9d5ed1d061a2e5  opsd-qwen3-1p7b-step100-hmmt25.tar.gz
71ba056f5432469488a1459a0d33dff92e9e529e77fb9eb74e6464999892cc37  opsd-qwen3-1p7b-step150-aime25.tar.gz
86f2b84723773730ce556cfe9fbe655250d1f8104f7325c192284f721f6e197f  opsd-qwen3-1p7b-step150-hmmt25.tar.gz
35228e00a6f75ab65468a4f78e14090091edd6022d3aebb02602bf9943b959e3  opsd-qwen3-1p7b-step200-aime25.tar.gz
53773439aa101efebf4db4be66b5887a47db1ff07ee6faa006052ccadc05c093  opsd-qwen3-1p7b-step200-hmmt25.tar.gz
0d7427c752c0c0eb5fa3273b6d3554b195e4cb723b091374a0783bf19111ccf7  opsd-qwen3-1p7b-step200-aime26.tar.gz
f5509904c72a8b4a5d0ec62cbfd6d3faa060d644d62c703530c73d82d63bc0ac  opsd-qwen3-4b-step50-aime25.tar.gz
d47fef7cd1ce64321b6cd76c9327ceb86409eceeeab7670fbff9bfab15a805ae  opsd-qwen3-4b-step50-hmmt25.tar.gz
d314eb8c9cbfbed65ceeebc241f62e607be05ee4ef95e3cec84e26da1a0f740e  opsd-qwen3-4b-step100-aime25.tar.gz
f584e03f617159c5b3d5127afa10a190a08579b68e52c27186bfd2c5a6469872  opsd-qwen3-4b-step100-hmmt25.tar.gz
a4d86e6f6fae4a917e81f30b33eaf81f5c1b98ea5dfd8fe566e243f4ccb2fe53  opsd-qwen3-4b-step150-aime25.tar.gz
b32e36e8498fdb4f46c15daaad361b04ddbdfe85a2a62125c40515ebc520adcc  opsd-qwen3-4b-step150-hmmt25.tar.gz
98966b01736034a241ec6f18e1fa87938e13ae0c2a7f916555136c275202e290  opsd-qwen3-4b-step200-aime25.tar.gz
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
