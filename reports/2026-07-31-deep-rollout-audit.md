# Deep rollout audit and method decision

Date: 2026-07-31

This report separates observed benchmark behavior, exact rollout evidence,
objective-level defects, and the resulting method decision. Scores are
problem-clustered over the available rollouts; a gain is not treated as robust
when its paired confidence interval includes zero.

## 1. Comparable 32k benchmark results

| Method | AIME 2025 base | AIME 2025 candidate | Delta | AIME 2026 base | AIME 2026 candidate | Delta |
|---|---:|---:|---:|---:|---:|---:|
| G4, common 6-rollout reconstruction | 237/360 | 250/360 | +3.61 pp | 225/360 | 231/360 | +1.67 pp |
| Legacy G1, common reconstruction | 237/360 | 232/360 | -1.39 pp | 225/360 | 245/360 | +5.56 pp |
| FRGD v1 | 66.39% | 65.83% | -0.56 pp | 62.78% | 66.94% | +4.17 pp |
| FRGD v2, 4k train rollouts | 66.39% | 63.89% | -2.50 pp | 62.78% | 68.61% | +5.83 pp |
| FiNOD, 32 steps | 62.22% | 66.67% | +4.44 pp | 68.33% | 66.11% | -2.22 pp |
| FiNOD, 5 steps | 62.22% | 65.00% | +2.78 pp | 68.33% | 62.78% | **-5.56 pp** |

The FiNOD 5-step AIME 2026 paired 95% problem-clustered interval is
[-11.11, -1.11] pp. It is the clearest negative result in the recent screen,
not noise that should be hidden by pooling benchmarks.

The strongest single historical score change is FRGD v2 on AIME 2026
(+5.83 pp, paired interval [2.78, 9.44] pp), but it loses 2.50 points on AIME
2025 and its teacher view explicitly contains the answer. It is therefore not
evidence for a clean answer-free method.

The standard 200-step Qwen3-4B OPSD trajectory shows the exposure failure
directly on AIME 2025: 66.39% (base), 64.72% (step 50), 56.39% (step 100),
44.17% (step 150), and 41.39% (step 200). Its step-200 AIME 2026 accuracy is
46.67%, with a corresponding HMMT collapse. More optimization is not a remedy.

## 2. Development-benchmark controls

| Method | AIME 2024 base | Candidate | Delta | Interpretation |
|---|---:|---:|---:|---|
| G1 answer-masked scaffold | 75.00% | 78.06% | +3.06 pp | weak positive Avg signal; majority/pass unchanged |
| G2 viability routing | 75.00% | 75.56% | +0.56 pp | effectively flat |
| G3 viability + entropy | 75.00% | 73.89% | -1.11 pp | negative |
| CH, no extra optimization | 77.50% | 75.83% | -1.67 pp | negative |
| CH, 1–12 steps | 77.50% | 77.50% | 0 | flat |
| CH, 50 steps | 77.50% | 71.67% | -5.83 pp | exposure collapse |
| Fluid context | 77.50% | 74.17% | -3.33 pp | negative |
| FiNOD, 5 steps | 73.89% | 73.89% | 0 | Avg flat; majority/pass +3.33 pp |

The zero-scale LoRA control ran the same LoRA/Punica inference path while
mathematically multiplying the adapter contribution by zero. All 180 paired
AIME 2024 responses were byte-identical to the no-LoRA baseline: zero answer,
length, finish-reason, or formatting changes and zero correctness flips.
Avg@6 remained 73.89%, Maj@6 76.67%, and Pass@6 86.67%. Consequently, the
historical treatment effects are not explained by merely enabling the adapter
runtime path.

## 3. What the exact rollouts show

### 3.1 The universal flip signature is basin selection

Across G1, G4, CH, FRGD, and FiNOD:

- correct flips usually terminate roughly 2k–8k tokens earlier and remove
  repeated loops;
- wrong flips often run 1k–10k tokens longer and hit the cap;
- however, shortening itself is not causal evidence of improved reasoning:
  a correct solver naturally stops after finding and checking the result;
- several candidate rollouts are shorter because they commit early to an
  unsupported assertion.

The learned change is best described as a shift in the probability of entering
particular reasoning basins, not acquisition of a general new mathematical
skill.

This signature survives an exact same-seed feature audit of all paired
rollouts for FiNOD-5, FiNOD-32, and FRGD-v2 on both locked benchmarks:

| Pair | Wrong→correct token delta | Wrong→correct cap delta | Correct→wrong token delta | Correct→wrong cap delta |
|---|---:|---:|---:|---:|
| FiNOD-5, AIME 2025 | -3,741 | -0.091 | +6,580 | +0.188 |
| FiNOD-5, AIME 2026 | -8,022 | -0.400 | +3,796 | +0.133 |
| FiNOD-32, AIME 2025 | -6,656 | -0.217 | +5,608 | +0.214 |
| FiNOD-32, AIME 2026 | -6,501 | -0.273 | +1,019 | +0.067 |
| FRGD-v2, AIME 2025 | -7,045 | -0.107 | +4,853 | +0.216 |
| FRGD-v2, AIME 2026 | -6,128 | -0.139 | +7,536 | +0.200 |

Successful flips also remove roughly 14–35 backtracking markers on average;
regressions add roughly 6–24. Three-gram repetition moves in the same
direction. These are descriptive consequences of the selected basin, not a
license to optimize token count or suppress checking.

### 3.2 Exact AIME 2026 FiNOD regressions

- Problem 10: base rollouts checked both rotations and the side condition and
  reached 156. A candidate accepted an incorrect polygon/shoelace area of 21
  even while calling it implausibly small.
- Problem 26: a base route factored the polynomial into roots
  \(16,3\pm4i\) and obtained 132. Candidate routes capped, optimized an
  unrelated quartic to 285, or guessed 270.
- Problem 28: a base route used
  \(4040=2^3\cdot5\cdot101\) and obtained 107. A candidate hand-waved
  \(4040\approx2^{12}\) and guessed 12.

Positive flips existed, but they were equally basin-specific:

- Problem 11 fixed grid-degree accounting and reached 896.
- Problem 23 corrected an integer scaling error from 1225 to 245.
- Problem 27 found exact coordinates and changed \(175/48\) to 223.

This is why a single guide can improve one benchmark and damage another: it
amplifies whichever route it makes locally attractive, whether or not that
route is sound.

### 3.3 Difficulty stratification

Using the untouched rollout success rate to define easy
(\(\geq2/3\)), medium (nonzero but below \(2/3\)), and hard (zero), the
aggregate correct-rollout changes are:

| Method | Benchmark | Easy | Medium | Hard |
|---|---|---:|---:|---:|
| FiNOD-5 | AIME 2025 | -7 | +12 | +1 |
| FiNOD-5 | AIME 2026 | -6 | -4 | 0 |
| FiNOD-32 | AIME 2025 | -2 | +11 | 0 |
| FiNOD-32 | AIME 2026 | -3 | -1 | 0 |
| FRGD-v2 | AIME 2025 | -7 | -2 | 0 |
| FRGD-v2 | AIME 2026 | +4 | +15 | +2 |

The clean methods almost never create success on a problem for which all base
rollouts fail. They redistribute probability among existing capabilities,
with the largest movement on medium problems, while sometimes taxing easy
ones. This sets the realistic aim for answer-free guidance: preserve easy
problems and improve route selection on medium problems. A pure guidance
method should not claim to manufacture missing mathematical knowledge.

### 3.4 Guide quality is the immediate failure mode

The accepted legacy cache has 1,385 graphs, averaging 2.386 forks and 7.817
actions. The viability labels are generated priors rather than empirical
outcomes. Only 32.5% of descriptions contain a problem-specific numeral or
entity and 24.4% contain even a basic mathematical marker.

Exact failures include:

- recommending enumeration of every divisor for a 25-prime-divisor
  probability problem instead of exploiting the product factorization;
- assuming a rotated ellipse's center-to-\(x\)-axis distance is its semiminor
  axis;
- steering a hexagon problem toward parsing Asymptote label coordinates;
- inserting incorrect inclusion-exclusion templates;
- generic instructions such as “calculate and check” that contribute style
  but little problem information.

All recorded AMC/AIME and AoPS training rollouts hit the 4k cap. Across the
128 recorded FiNOD training completions, only 31 closed the thinking segment
and 37 emitted a boxed result.

True guide text does imprint the completion: its lexical overlap with the first
6k completion tokens is 0.1366 versus 0.0485 for a shuffled guide, and the true
guide wins 99.2% of comparisons. The problem is not that the teacher ignores
the guide. The problem is that imprint strength does not predict completion or
correctness.

### 3.5 Domain imbalance explains part of the cross-benchmark instability

The original 1,024-example FiNOD subset was selected by data source and
reference-response length, not by mathematical domain. It contained 728
`olympiads` rows, 165 Chinese-contest rows, 91 AoPS rows, and only 21
AMC/AIME rows. A problem-text diagnostic found 371 geometry-like rows but only
71 number-theory-like and 76 combinatorics-like rows; the remaining rows were
mostly unclassified or algebra/sequence/probability. This was not a balanced
AIME curriculum.

AIME 2025 has official MathArena multi-label types: 9 algebra, 8 geometry,
6 number theory, and 9 combinatorics problems. AIME 2026 lacks type metadata
in the pinned revision, so its 30 public problem statements were manually
assigned the same taxonomy: approximately 6 algebra, 10 geometry, 5 number
theory, and 12 combinatorics labels, including three cross-domain problems.
Per-domain paired rollout changes were:

| Method | Benchmark | Algebra | Geometry | Number theory | Combinatorics |
|---|---|---:|---:|---:|---:|
| G4 | AIME 2025 | -0.93 pp | +6.25 pp | -1.39 pp | +6.48 pp |
| G4 | AIME 2026 | +1.39 pp | +6.67 pp | 0.00 pp | +0.69 pp |
| Legacy G1 | AIME 2025 | -6.48 pp | +1.04 pp | -1.39 pp | 0.00 pp |
| Legacy G1 | AIME 2026 | +2.78 pp | +5.00 pp | +1.67 pp | +7.64 pp |
| FRGD v2 | AIME 2025 | -5.56 pp | -4.17 pp | 0.00 pp | +0.93 pp |
| FRGD v2 | AIME 2026 | +5.56 pp | +5.83 pp | +1.67 pp | +6.94 pp |
| FiNOD, 32 steps | AIME 2025 | -9.26 pp | +8.33 pp | 0.00 pp | +14.81 pp |
| FiNOD, 32 steps | AIME 2026 | 0.00 pp | -1.67 pp | -3.33 pp | -4.17 pp |
| FiNOD, 5 steps | AIME 2025 | -9.26 pp | +2.08 pp | -2.78 pp | +16.67 pp |
| FiNOD, 5 steps | AIME 2026 | -11.11 pp | 0.00 pp | -3.33 pp | -8.33 pp |

Multi-label AIME problems contribute to each applicable row. These are
descriptive subgroup estimates, not independent significance tests. The
consistent fact is that aggregate movement hides large, opposing domain
effects. In particular, FiNOD's AIME 2025 gain came from geometry and
combinatorics while algebra sharply regressed.

## 4. Objective and implementation defects

The training telemetry rules out a simple exploding-gradient account:

| Run | Logged loss start→end | Mean/max grad norm | Rollout cap rate | Boxed |
|---|---:|---:|---:|---:|
| G1 | -0.0085→-0.0113 | 0.136/0.225 | 99.2% | 7/200 |
| G2 | 0.0116→-0.0025 | 0.155/0.281 | 99.6% | — |
| G3 | 0.0158→-0.0020 | 0.159/0.290 | 99.5% | — |
| G4 | 0.0312→0.0084 | 0.0637/0.131 | 90.0% | 82/300 |
| CH-50 | 0.0006→-0.0072 | 0.0266/0.0501 | 72.7% | 90/200 |
| FRGD-v1 | 0.0049→0.0052 | 0.0330/0.0502 | 99.9% | 0/128 |
| FRGD-v2 | 0.0041→0.0039 | 0.0254/0.0835 | 79.5% | 43/128 |
| FiNOD-5 | 0.0005→0.0004 | 0.0262/0.0317 | 85.6% | 9/20 |
| FiNOD-32 | 0.0005→0.0005 | 0.0244/0.0358 | 83.8% | 37/128 |

The registered max-gradient norm is 0.1 for the recent runs; it is rarely
active in FiNOD. The key failures are target quality, objective validity,
truncation, and cumulative direction drift—not an uncontrolled gradient norm.

### 4.1 Historical G1–G4 / CH divergence

The historical “forward KL” clipped each vocabulary-element KL summand at
0.05 before summing. Individual summands can be negative. Clipping only their
positive side destroys KL non-negativity, which explains negative reported
losses. Those loss curves cannot be interpreted as a decreasing divergence.

CH teacher dossiers also contain the literal trusted final answer and full
reference solution. Fluid routing repaired the sign but not the graph
semantics, and its benchmark result was worse.

### 4.2 FRGD leakage

FRGD constructs a teacher view stating the final answer and derives a Fisher
projection against it. Removing one local linear component does not remove
nonlinear semantic information about that answer. FRGD is therefore an
interesting privileged diagnostic, but not a clean solution to answer
leakage.

### 4.3 Original FiNOD leakage and anchor drift

The original FiNOD collator retained an answer-control view. A one-sided
projection removed positive answer alignment but deliberately preserved
anti-aligned answer signal. The signed variant avoids that exact asymmetry but
still processes an answer-conditioned distribution.

The original target also used the current detached student distribution as
its trust anchor. That bounds each optimizer step but does not bound the final
model from the original deploy policy.

Finally, the purported base teacher prompt was not the student/deploy prompt:
it contained an auxiliary-context wrapper and “continue/check/revise” prose.
Consequently, even a zero residual could distill prompt style.

### 4.4 Dose drift

The 5-step and 32-step FiNOD adapters are not the same update at different
scale:

- \(\|\Delta W_5\|_F=0.08564\)
- \(\|\Delta W_{32}\|_F=0.23791\)
- norm ratio 2.778
- cosine similarity 0.524
- \(\|\Delta W_{32}-\Delta W_5\|/\|\Delta W_{32}\|=0.867\)

The later 27 steps introduce a large new direction. This matches the broader
observation that long OPSD exposure destroys initially easy problems.

### 4.5 Evaluation-path instability

Changing six to twelve samples per problem changes flattened request indices,
shard membership, and batch grouping. Even with the same nominal seeds and
prompt hashes, autoregressive numerical divergence changes outputs:

- AIME 2025 first-six accuracy changed by -7.78 pp;
- AIME 2026 changed by +3.89 pp;
- only one differing AIME 2025 output required more than 32k tokens, so the
  token cap is not the explanation.

Every future comparison must keep sample count, flattening, shard assignment,
runtime, and adapter path fixed. The zero-scale LoRA control measures the
remaining no-op runtime displacement.

## 5. Fisher specificity diagnostic

On 12 exact rollout prefixes and 48 positions:

- true-guide Fisher energy: 0.01090;
- true versus generic-control cosine: 0.6194;
- true versus shuffled-guide cosine: 0.6918;
- generic projection retains 29.9% of energy;
- shuffled-guide projection retains 24.3%;
- after generic style removal, true versus shuffled cosine remains 0.5374.

At the start of a completion, all guide directions are nearly collinear
(cosines 0.94–0.96), reflecting prompt structure. At one-third of successful
rollouts the guide supports the actually generated token, whereas late in
unfinished rollouts it strongly opposes it. This supports two decisions:

1. train only the early prefix;
2. use matched shuffled plans as a contrast, not a fixed generic style prompt.

## 6. Method decision: Fisher Consensus Guidance

For a problem \(x\), let \(p_0(\cdot\mid h)\) be the frozen base model under
the exact deploy prompt at rollout prefix \(h\). Generate \(K=3\) concise plans
\(g_k\) independently from \(x\) alone. Pair each with a source- and
length-matched plan \(c_k\) generated for another problem. No answer,
reference solution, reward, verifier, or correctness label is available.

At a selected early-prefix token, construct the centered contrastive tangent

\[
u_k =
\operatorname{Center}_{p_0}\left[
z_0(x,g_k,h)-z_0(x,c_k,h)
\right].
\]

The Fisher inner product is

\[
\langle a,b\rangle_{F,p_0}
=\sum_v p_0(v\mid h)a_vb_v.
\]

Let

\[
\bar u=\frac1K\sum_k u_k,\qquad
A=
\frac{\|\bar u\|_{F,p_0}^2}
{\frac1K\sum_k\|u_k\|_{F,p_0}^2+\epsilon}.
\]

By Jensen's inequality, \(A\in[0,1]\). Identical contrastive directions have
\(A=1\); mutually cancelling directions have \(A=0\). The retained direction
is simply

\[
r=A\bar u.
\]

The detached target is an exponential tilt of the frozen deploy anchor:

\[
q_\eta(v\mid h)
\propto p_0(v\mid h)\exp(\eta r_v).
\]

Per-token bisection chooses the largest \(\eta\le 0.25\) satisfying both

\[
D_{\mathrm{KL}}(q_\eta\|p_0)\le0.01,\qquad
D_{\mathrm{KL}}(p_0\|q_\eta)\le0.01.
\]

The student minimizes the ordinary nonnegative
\(D_{\mathrm{KL}}(q_\eta\|p_\theta)\) on 32 positions sampled within the first
1,024 rollout tokens.

This is intentionally one mechanism, not a mixture:

- contrast removes prompt/style effects and irrelevant-plan influence;
- Fisher agreement suppresses unstable plan-specific directions;
- the frozen anchor prevents cumulative trust-region drift;
- the early-prefix restriction targets route selection before wrong basins
  become self-reinforcing.

## 7. Cache and leakage contract

The new builder:

- receives only `question`, `data_source`, and source index;
- uses PyArrow column projection so the response column is never materialized
  in the builder process;
- creates three separately sampled plans with different seeds and planning
  lenses;
- assigns one primary standard AIME domain from problem-only structural cues,
  with a separately pinned problem-only model label used only as fallback;
- rejects boxed expressions, answer/result claims, prompt injection, explicit
  numeric equalities, and numerals not already present in the problem;
- stores no answer or reference field;
- cryptographically binds every saved problem and the complete records file;
- constructs negative controls by deterministic source/length-matched
  derangements within the same mathematical domain.

The Fisher trainer and selector now also use the problem-only PyArrow
projection. Unlike the inherited FiNOD launcher, they never materialize a
reference response or use reference-response length for partitioning. The
screen is exactly balanced across algebra, geometry, number theory, and
combinatorics, then source- and question-length-stratified within each domain.

The training row contains exactly:

`problem`, `solution` (guide zero for upstream compatibility),
`fisher_guides`, `fisher_controls`, and `fisher_source_index`.

The published schema-v3 cache at source commit `6c67ead` was rebuilt from
1,536 problem-only candidates and contains 843 validated examples:
221 algebra, 204 combinatorics, 272 geometry, and 146 number theory. The
fixed 5% problem-hash holdout leaves 803 eligible records. The exact
192-example screen contains 48 examples per domain, 28 AMC/AIME and 164 AoPS
problems, with every source/domain cell split across four question-length
quartiles. Its 576 plans have median 120 words and maximum pairwise trigram
Jaccard similarity 0.429. Exact control reconstruction found no self-control,
duplicate-control, source mismatch, or domain mismatch. The exact loader
revalidated every problem hash, plan constraint, schema field, and records-file
SHA-256 before the runtime smoke.

## 8. Screen and promotion rule

The first screen uses:

- Qwen3-4B at immutable revision `1cfa9a7`;
- 192 AMC/AIME + AoPS examples selected as exactly 48 examples from each of
  algebra, geometry, number theory, and combinatorics;
- five optimizer steps, effective batch 32;
- fused PyTorch AdamW, learning rate \(5\times10^{-6}\), betas
  \((0.9,0.999)\), epsilon \(10^{-8}\), zero weight decay;
- linear decay over five steps with no warmup;
- LoRA rank 64, alpha 128, all attention and MLP projections;
- gradient norm cap 0.1;
- 1,024-token on-policy training rollouts, exactly matching the supervised
  prefix (the runtime smoke showed every 4,096-token rollout capped, so the
  extra 3,072 tokens were unused generation);
- 32 Fisher positions in the first 1,024 tokens;
- three guide/control pairs;
- \(\eta_{\max}=0.25\), two-sided KL cap 0.01.

The one-step DP8/GA4 runtime smoke completed at job `17358`. Across its four
accumulation batches, Fisher loss was finite and nonzero
(\(4.69\times10^{-4}\) to \(7.76\times10^{-4}\)); mean loss was
\(6.21\times10^{-4}\). Agreement ranged from 0.512 to 0.576, the
noncollapsed-token fraction ranged from 0.668 to 0.727, and only 1.95% to
3.13% of tokens required trust-region clipping. Every recorded maximum
two-sided target KL was at most 0.01. At initialization, student loss divided
by frozen-anchor target KL was 1.000008 to 1.000011, the required identity for
an exact base-initialized student. The optimizer saw gradient norm 0.0362
under the 0.1 cap and produced adapter update norm 0.0404
(\(5.51\times10^{-4}\) relative to parameter norm). Peak GPU memory was
27.5 GiB, leaving substantial A100 headroom.

Promotion requires all of the following:

1. finite, nonnegative loss and nonzero gradients;
2. noncollapsed consensus energy and agreement meaningfully below one;
3. maximum observed target KL at or below 0.01;
4. improvement over both the no-LoRA baseline and zero-scale LoRA runtime
   control on AIME 2024 under the exact same six-rollout 32k protocol;
5. no material increase in capped completions;
6. exact flip inspection shows better checks rather than earlier unsupported
   commitment.

Only after that screen should a representative 1,024-example run and locked
AIME 2025/2026 evaluation be spent.

## 9. Preliminary novelty boundary

The closest current papers solve materially different problems:

- [GeoSD](https://arxiv.org/abs/2607.06855) assumes a privileged hint or full
  solution, attenuates teacher pulls with Hellinger overlap, and adds a
  Fisher–Rao proximal/natural-gradient treatment of drift.
- [Multi-Rollout OPD](https://arxiv.org/abs/2605.12652) conditions on
  verifier-labelled peer successes and failures.
- [MOPD](https://arxiv.org/abs/2606.30406) integrates capabilities from
  separately RL-trained domain teachers.
- [KAT](https://arxiv.org/abs/2606.09471) detects persistent low-KL agreement
  traps and terminates weak rollout supervision.
- [Classifier-Free Guidance for language models](https://arxiv.org/abs/2306.17806)
  already establishes positive-minus-negative prompt logit arithmetic at
  inference, and
  [Distillation Contrastive Decoding](https://arxiv.org/abs/2402.14874)
  combines contrastive prompting with distillation. Therefore, subtracting a
  control prompt by itself is not the novelty.
- Standard multi-teacher and contrastive distillation aggregate class
  predictions or hidden representations; they do not construct matched
  problem-plan/control-plan tangents under a frozen next-token Fisher metric.

The current search did not find the exact combination of problem-only
independent plans, matched shuffled-plan contrasts, the
\(\|\mathbb{E}u\|_F^2/\mathbb{E}\|u\|_F^2\) coherence shrinkage, and a
two-sided KL-bounded exponential tilt on on-policy reasoning prefixes.
That supports a plausible novelty claim, not a proof of novelty. An ICLR claim
should be made only after broader citation-chain review and, more importantly,
after the screen demonstrates a real cross-benchmark effect.

The defensible technical novelty, if the experiment works, is thus the
label-free Fisher coherence estimator and its use to decide how much
contrastive procedural guidance is locally safe to distill—not generic
positive/negative logit subtraction.
