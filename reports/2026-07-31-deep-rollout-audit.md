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

The first five-step screen, job `17375`, exposed a distinct optimization
failure despite passing all target-construction checks. Step-one loss equalled
the frozen-anchor target KL as required, but on later unseen batches mean
student-to-target loss was 1.95–2.40 times the corresponding anchor-to-target
loss. Gradients remained small (0.042–0.052) and never reached the 0.1 clip.
Thus neither exploding gradients nor target KL caused the failure. The first
AdamW update moved the student roughly one target-radius away from the base,
but gave no measurable cross-problem alignment; on a fresh problem, its
deviation added approximately orthogonally to the new guidance direction.
This is precisely the distinction between bounding each detached target and
bounding the learned policy itself.

The next diagnostic, job `17377`, made only one optimizer change: learning
rate \(10^{-6}\), with the same target, data, effective batch, and linear
schedule. It ran 12 steps (two passes over 192 examples) and additionally
logged
student-to-anchor KL plus the Fisher three-point alignment gain
\[
D_{\mathrm{KL}}(q\|p_0)+D_{\mathrm{KL}}(p_0\|p_\theta)
-D_{\mathrm{KL}}(q\|p_\theta).
\]
Positive gain means the learned displacement aligns with guidance rather than
merely increasing distance from the base.

The lower-rate run completed cleanly but rejected the learning-rate
hypothesis. Its first-epoch mean student-to-target/anchor-to-target ratio was
1.97 and its second-epoch ratio worsened to 2.16. Mean alignment gain was
\(-3.24\times10^{-5}\) then \(-2.38\times10^{-5}\), while the cosine proxy
remained near zero. Gradients stayed below the clip and the final per-step
adapter update norm decayed to \(2.31\times10^{-4}\). The residual is therefore
nonzero and numerically safe, but it is too weak and cross-problem-orthogonal
for default fused AdamW's \(10^{-8}\) epsilon: the optimizer behaves close to
a coordinatewise sign update on many low-second-moment LoRA coordinates.

The next mechanism screen keeps the method, exact balanced subset, learning
rate, batch, and schedule fixed while raising only Adam epsilon to
\(10^{-4}\). This makes the update proportional to the small measured
residual instead of normalizing it almost entirely by its own tiny second
moment. A one-step calibration must first show an appropriately smaller
adapter displacement; only then is the full 12-step screen justified.

The one-step calibration, job `17381`, reduced adapter update norm from
0.00812 to 0.000354 (23-fold) without eliminating the Fisher loss. However,
the complete 12-step control, job `17382`, showed that parameter-space norm
was not the governing failure. Epoch-one and epoch-two mean target-loss
ratios were 1.91 and 2.15, with student-anchor KL again approximately equal
to one target radius and alignment near zero. A small number of
high-sensitivity LoRA directions can therefore create the same
function-space displacement despite a much smaller Euclidean parameter norm.

The next screen adds the missing policy-space constraint directly:
\[
\mathcal L_{\mathrm{prox}}
=D_{\mathrm{KL}}(q\|p_\theta)
+\lambda D_{\mathrm{KL}}(p_0\|p_\theta),\qquad \lambda=1.
\]
The second term uses the already-computed frozen deploy distribution at the
same on-policy prefix positions. It accesses no answer, label, verifier, or
reference solution. Target-fitting loss, frozen-policy KL, and the combined
optimization objective are logged separately so an apparently improving
regularized scalar cannot hide failure to learn the guidance direction.

Job `17385` showed that the proximal geometry helped but was insufficient at
the original target scale. Relative to its optimizer-matched control, its
mean target-loss ratio improved from 1.91 to 1.895 in epoch one and from 2.153
to 2.106 in epoch two. Alignment gain became positive
(\(5.68\times10^{-5}\), then \(3.32\times10^{-5}\)), but remained only about
one tenth of student-anchor KL, so target loss never beat the frozen anchor.
The checkpoint is not promoted.

The next one-epoch diagnostic raises only the exponential-tilt ceiling
\(\eta\) from 0.25 to 1.0. The two-sided per-token KL cap remains exactly
0.01, and the \(\lambda=1\) frozen-policy proximal remains active. This
distinguishes a sub-noise Fisher target from a fundamentally
non-transferable direction without relaxing the function-space safety bound.

The strong-target job `17387` completed safely and reduced the mean
target-loss ratio to 1.254, but did not cross below one. Only 14.3% of target
tokens hit the two-sided 0.01 KL cap. Thus target scale was part of the
optimization mismatch, but not the source of transfer.

Job `17389` then tested leave-one-out Fisher-edge boosting across the eight
problems in each global data-parallel microbatch. The edge was nondegenerate:
54.7% of examples were active and positive edge averaged 0.0495. Nevertheless
the mean ratio worsened to 1.277 and alignment became slightly negative. A
shared vocabulary score is therefore not a useful proxy for parameter-space
transfer in this setting.

These failures expose a more basic cancellation error. Both a problem's
answer-free plans and unrelated control plans contain generic contest-solving
procedure. Guide-minus-control subtraction removes that shared procedural
component and leaves a problem-specific residual. The next screen uses the
Fisher barycenter of positive plans relative to the frozen base:
\[
u_k=(z_k^+-z_0)-\mathbb E_{p_0}[z_k^+-z_0].
\]
Three-plan agreement, the strong target ceiling, two-sided KL cap, and
frozen-policy proximal remain unchanged. Controls remain an audited placebo
but cannot affect the barycenter target.

Job `17391` validated the diagnosis. Positive-plan agreement rose to 0.725,
post-initial Fisher alignment gain became \(2.56\times10^{-4}\), and alignment
cosine became 0.0477. The post-initial target-loss ratio fell to 1.154, much
closer to transfer than any residual target, but remained above one because
student-anchor KL was \(6.29\times10^{-4}\). In the local quadratic regime,
alignment gain scales linearly with displacement while anchor KL scales
quadratically. Their ratio predicts that reducing displacement below about
0.41 of its current size should make gain exceed drift. The next screen
therefore changes only the frozen-policy proximal weight from 1 to 4.

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

### 8.6 Positive-plan barycenter task screen

The six-step positive-plan barycenter checkpoint from job `17391` was
evaluated on all 30 AIME-2024 problems with six paired rollouts per problem
and a 32,768-token generation limit. Against the exact same-seed frozen-base
run:

| Metric | Frozen base | Barycenter | Delta |
|---|---:|---:|---:|
| rollout accuracy | 133/180 (73.89%) | 137/180 (76.11%) | +2.22 pp |
| majority accuracy | 23/30 (76.67%) | 25/30 (83.33%) | +6.67 pp |
| pass@6 | 26/30 (86.67%) | 25/30 (83.33%) | -3.33 pp |
| 32k cutoffs | 10/180 | 14/180 | +4 |
| mean output tokens | 14,582 | 15,518 | +936 |

The rollout-level paired table contains 128 both-correct, 38 both-wrong,
nine wrong-to-correct, and six correct-to-wrong outcomes. The bootstrap
95% interval for the average delta is `[-1.11 pp, +6.67 pp]`, so the result is
a positive screen rather than conclusive evidence.

The flip audit sharply localizes the unsafe behavior. Correct-to-wrong flips
grew by 17,339 tokens on average and account for all four newly introduced
cutoffs. Wrong-to-correct flips shortened by 4,840 tokens on average and
removed three cutoffs. The strongest gain was AIME I problem 13, where the
candidate enumerated all Hensel lifts and improved from 1/6 to 4/6. The
pass@6 loss was AIME II problem 8, where the sole correct geometric derivation
of 127 changed to an unsupported 7 and the other candidate paths repeatedly
restarted the geometry.

The method therefore improves the probability of entering some valid
reasoning basins, but its full learned displacement also pushes a minority of
previously valid paths into long indecisive basins. It fails the preregistered
no-material-cutoff-increase gate at scale one.

The next one-variable screen uses the same adapter weights at LoRA scale
`3/8`, corresponding to `lora_alpha=48` rather than 128. This directly tests
the earlier Fisher three-point prediction that alignment should dominate
quadratic anchor drift below roughly 0.41 of the learned displacement.

The \(3/8\) evaluation completed in job `17413`:

| Metric | Frozen base | \(3/8\) barycenter | Delta |
|---|---:|---:|---:|
| rollout accuracy | 133/180 (73.89%) | 134/180 (74.44%) | +0.56 pp |
| majority accuracy | 23/30 (76.67%) | 24/30 (80.00%) | +3.33 pp |
| pass@6 | 26/30 (86.67%) | 25/30 (83.33%) | -3.33 pp |
| 32k cutoffs | 10/180 | 12/180 | +2 |
| mean output tokens | 14,582 | 15,222 | +640 |

There were 11 improvements and 10 regressions. Improvements shortened by
4,983 tokens on average and removed three cutoffs; regressions grew by 9,673
tokens and introduced three cutoffs. The interpolation preserved and even
strengthened the AIME I problem 13 gain to 5/6, but the sole correct AIME II
problem 8 path still changed from 127 to the unsupported answer 7. It also
introduced new losses on AIME II problems 2, 5, 7, and 14.

Thus direct displacement shrinkage reduces aggregate drift but does not
remove the unstable basin-selection mechanism. It fails the pass@6 and cutoff
gates and is rejected. The next isolated screen shortens only the supervised
Fisher horizon from 1,024 to 512 tokens while preserving the 32k inference
budget.

Per the subsequent evaluation amendment, no further screen uses AIME-2024.
The prefix-512 and unscaled prefix-1,024 checkpoints are compared only on
AIME-2025 and AIME-2026 with six paired rollouts and the 32k cap.

Prefix-512 training job `17428` completed safely. All losses were finite and
nonzero, maximum two-sided target KL remained at or below 0.01, and the
adapter update completed. Its post-initial target-loss/frozen-anchor ratio
was 1.1575 versus 1.1540 for prefix 1,024. Alignment gain fell from
\(2.56\times10^{-4}\) to \(2.17\times10^{-4}\), alignment cosine from 0.0477
to 0.0178, and clipping rose from 18.65% to 23.38%. The shorter horizon is
therefore not better in the local target geometry; only the long-generation
task evaluations can support its basin-selection hypothesis.

### 8.7 Allowed-benchmark barycenter screens

The fixed comparison baseline has 112/180 correct rollouts on AIME-2025 and
123/180 on AIME-2026. Each entry below uses the same six paired seeds per
problem and the same 32,768-token limit.

| Screen | AIME-2025 Avg@6 | Maj@6 | Pass@6 | Cutoffs | AIME-2026 Avg@6 | Maj@6 | Pass@6 | Cutoffs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| frozen base | 112/180 | 22/30 | 26/30 | 15 | 123/180 | 22/30 | 26/30 | 14 |
| prefix-512 sequential (S10) | 114/180 | 21/30 | 24/30 | 19 | 124/180 | 24/30 | 25/30 | 14 |
| full-prefix global batch (S11) | 126/180 | 24/30 | 24/30 | 11 | 112/180 | 23/30 | 25/30 | 12 |
| prefix-512 global batch (S12) | 110/180 | 21/30 | 22/30 | 18 | 117/180 | 22/30 | 25/30 | 11 |
| entropy-neutral prefix-512 (S13) | 113/180 | 24/30 | 26/30 | 23 | 113/180 | 21/30 | 25/30 | 13 |

S10 is the only completed recent screen with a positive average-score change
on both years, but it loses majority or pass coverage and adds four AIME-2025
cutoffs. S11 demonstrates that global optimizer aggregation can produce a
large AIME-2025 gain while reversing sharply on AIME-2026. Combining the two
axes in S12 does not repair the reversal.

S13 removes the consensus component parallel to the frozen entropy gradient.
Its six-step training run was finite and nonzero, retained 44.8% of consensus
energy, reduced the residual first-order entropy alignment below
\(3.5\times10^{-9}\), and respected the 0.01 two-sided target-KL cap. However,
the finite exponential target still increased entropy by 0.00324 on average.
On AIME-2025, S13 gains one correct rollout and two majority-correct problems
while preserving pass@6, but adds eight cutoffs. All eight additional cutoffs
occur in correct-to-wrong transitions; wrong-to-correct transitions remove
three cutoffs. Problems 21 and 29 each fall from 5/6 to 3/6 and add five
cutoffs together, while problem 27 rises from 2/6 to 6/6 and removes two.
This is direct evidence that first-order entropy neutrality retains useful
route steering but does not control the finite retraction's basin dispersion.

The AIME-2026 evaluation completed in job `17525`, with paired comparison job
`17526`. Average accuracy fell from 123/180 (68.33%) to 113/180 (62.78%), a
-5.56-point change with paired problem-clustered interval
[-12.22, +0.56] points. Majority fell from 22/30 to 21/30 and pass@6 from
26/30 to 25/30. Cutoffs decreased from 14 to 13, so this second failure is not
explained by length truncation. There were 18 correct-to-wrong transitions
and eight wrong-to-correct transitions. Problem 10 fell from 5/6 to 1/6 by
selecting the invalid 103-area rotation instead of the checked 156 route;
problem 26 fell from 4/6 to 1/6 as candidate paths abandoned the exact
132-root argument for unsupported 252, 300, or 852 branches. The sole correct
baseline path on problem 17 also disappeared. Conversely, problem 27 rose
from 1/6 to 3/6 by recovering the exact 223 coordinate derivation.

S13 is therefore rejected for two separable reasons: finite exponential
curvature creates long dispersion on AIME-2025, while the unconstrained
finite redistribution can also concentrate on a confident but invalid route
on AIME-2026. Any next retraction must preserve the useful Fisher tangent
while controlling the finite move's self-information, not merely its
first-order entropy derivative.

### 8.8 Self-information mixture retraction (S14)

S14 keeps S13's guide construction and entropy-neutral Fisher tangent but
replaces the finite exponential map with

\[
q_\alpha(a)=p_0(a)(1+\alpha r_\perp(a)).
\]

Because both \(\mathbb E_{p_0}r_\perp=0\) and
\(\mathbb E_{p_0}[r_\perp\log p_0]=0\), this mixture-geodesic target is
normalized and preserves frozen cross entropy exactly. Consequently,

\[
H(q_\alpha)=H(p_0)-D_{\mathrm{KL}}(q_\alpha\Vert p_0),
\]

so the finite target cannot increase entropy or collectively migrate toward
lower-probability frozen tokens. A positivity bound on \(\alpha\) precedes the
unchanged two-sided 0.01 KL bisection.

The implementation commit `a7ac99f` passes 315 tests plus 68 subtests and
five production-shaped 151,936-vocabulary float32 stress cases. A one-step
eight-A100 smoke, job `17537`, completed in 2:40 with finite nonzero loss,
gradient norm 0.0442, 26.0 GiB peak memory, and a nonzero adapter. The full
six-step equal-domain run, job `17540`, completed in 6:27. Its training audit
shows:

- 24/24 finite positive Fisher events and maximum target KL 0.01;
- mean agreement 0.7250 and retained tangent energy 0.4469, matching S13;
- mean target entropy change -0.001157;
- maximum mean absolute cross-entropy residual \(1.35\times10^{-8}\);
- positivity limiting on 52.2% of target positions, while mean effective
  targets remain nonzero;
- post-initial alignment gain \(1.08\times10^{-4}\), cosine 0.0305, and
  target-loss/frozen-anchor ratio 1.453;
- gradient norms 0.0442--0.1159, with the registered 0.1 clip active once;
- all 252 LoRA-B tensors and all 70,778,880 entries nonzero, adapter L2
  0.000910, and adapter SHA-256 `dad16eaa6d6cfc353bc6909d4f8a12590cc80c20fb9230aa9356da83b7dbf236`.

The exact conservation law is working, but the positivity boundary makes the
target roughly half as strong while the learned student drift remains near
S13's scale. Per the preregistered design, AIME-2026 was the first strict task
screen. Generation job `17545` completed in 34:02 and paired comparison job
`17546` completed in eight seconds. Average accuracy fell from 123/180
(68.33%) to 118/180 (65.56%), a -2.78-point change with paired
problem-clustered interval [-7.78, +2.78] points. Majority stayed 22/30, but
pass@6 fell from 26/30 to 24/30 and cutoffs rose from 14 to 16. There were 18
correct-to-wrong and 13 wrong-to-correct transitions.

The exact flips reject a length-only explanation. Problem 11 improved from
2/6 to 5/6 by recovering the degree-weighted checkerboard extremum 3,896;
problem 18 improved from 1/6 to 3/6 by deriving the correct coordinate-area
congruence and 503 admissible lengths. Conversely, problem 9 lost its only
correct 9/20 conditional-counting path and replaced it with an incorrect
12/125 case split; problem 17 lost its sole correct 243 path for an unsupported
perfect-square guess; and problem 26 fell from 4/6 to 2/6. In the latter,
candidate paths abandoned an exact factorization and enumeration proving
`n=132` for invalid floor or speculative quartic arguments yielding 285 or
300. Problems 9 and 17 account for both pass@6 losses. Newly capped degraded
paths occur on problems 12 and 24, while problem 29 adds two more cutoffs.

S14 is therefore rejected and receives no AIME-2025 evaluation. Exact
self-information conservation prevents aggregate migration toward unlikely
frozen tokens, but it cannot distinguish checked reasoning from a confident
invalid route. The registered full-support I-projection contingency S15 is the
last isolated retraction test; if it fails, the target semantics/localization,
not the finite information geometry, becomes the leading cause.

### 8.9 Procedural-plan style audit

The selected training manifest was joined exactly to all 192 selected source
indices, preserving 48 examples per domain and yielding 576 actual guide
plans. Their builder prompt explicitly requests falsification checks and a
fallback. In the selected plans, 56.9% mention an alternative, 44.6% mention
verification, 28.8% contain `fallback`, 25.5% contain `critical check`, and
21.2% use the exact phrase `if this route fails`. This is not an unused-cache
artifact: the counts are over the plans consumed by S10--S14.

That semantic pattern matches the task evidence. Correct-to-wrong transitions
are commonly longer or end in speculative route changes, while improvements
often shorten after finding a decisive invariant. S14's AIME-2026 degraded
transitions add 4,945 tokens on average and three cutoffs; improved transitions
remove 6,316 tokens and two cutoffs. Retraction geometry can control how far a
plan-conditioned distribution moves, but cannot remove fallback/restart style
embedded in the direction itself. If S15 fails, the next one-variable screen
must localize or remove recovery-style guidance rather than add another
retraction constraint.

### 8.10 Full-support self-information I-projection (S15)

S15 replaces only the S13/S14 finite retraction with

\[
q_{\alpha,\beta}(a)\propto
p_0(a)\exp\{\alpha r_\perp(a)+\beta h(a)\},
\quad
h=\log p_0-\mathbb E_{p_0}\log p_0,
\]

where the scalar `beta` is solved so that
`E_q h = 0`. The moment is monotone with derivative `Var_q(h)`, giving a
unique root for nonuniform anchors. Since `r_perp` is Fisher-orthogonal to
`h`, `beta'(0)=0`; S15 has exactly the same local tangent as S13/S14 while
preserving frozen cross entropy and satisfying
`H(q)-H(p) = -KL(q||p)` at finite scale. Unlike S14 it has no positivity step
boundary.

Implementation commit `a938647` passes 318 tests plus 69 subtests. Production
float32 stress at vocabulary size 151,936 caught and fixed a target
renormalization error before cluster use; subsequent scale-1, scale-8, and
scale-25 cases respect the 0.01 two-sided KL cap with cross-entropy residuals
at or below `9.54e-7` and finite targets.

The eight-A100 one-step smoke, job `17556`, completed in 2:50. It produced
finite nonzero loss 0.0019, gradient norm 0.0542, four positive Fisher events,
maximum target KL 0.01, zero positivity limiting, maximum mean absolute
cross-entropy residual `1.17e-8`, and maximum entropy/KL identity residual
`9.78e-9`. Its LoRA-B update has 70,718,445/70,778,880 nonzero entries, L2
0.000420, maximum absolute value `8.75e-7`, and SHA-256
`68cc925a2ea33a208ca24efc0ead03528e7bcb2ef5f3e5f9e3c689370017cecd`.
The first six-step attempt, job `17560`, reached the final optimizer step but
hard-failed before checkpoint publication. One rank encountered a nearly
uniform frozen distribution whose entropy variance was below the projection
activity threshold; the implementation therefore treated it as exactly
uniform even though the finite tilted distribution had self-information
moment 0.00120, above the `3.74e-5` tolerance. No partial checkpoint is used.

The corrected implementation separates entropy-projection activity from
moment-solve activity using the range of centered self-information. It also
uses a scale-aware monotone bracket centered at zero for rare low-variance
roots. A dedicated near-uniform regression fails on the old implementation
and passes after the fix. The full suite now has 319 tests plus 69 subtests;
production-shaped near-uniform float32 stress at vocabulary size 151,936 has
zero reported cross-entropy residual, maximum target KL 0.01, and finite
multiplier 48.2. Fix commit `58ecd85` completed the clean six-step retry, job
`17562`, in 7:08; the failed directory is preserved separately and was never
resumed.

The retry audit has 24/24 finite positive Fisher events, maximum target KL
0.01, zero positivity limiting, maximum mean absolute cross-entropy residual
`2.08e-8`, and maximum entropy/KL identity residual `1.11e-8`. Mean target
entropy change is -0.001903. Post-initial target-loss/frozen-anchor ratio is
1.2484, alignment gain `1.92e-4`, and cosine 0.0230. Gradient norms are
0.0542--0.1438; the registered 0.1 cap is active on the final two steps. All
252 LoRA-B tensors and all 70,778,880 entries are nonzero, adapter L2 is
0.001060, maximum absolute value is `2.71e-6`, and SHA-256 is
`4a81f8031bb1487cf9b7901c36522fcad961953aae89f2eeb7703a138595d879`.
The exact 192-example selection remains 48 examples per domain with selection
SHA-256 `fdca7d7b30ed95e36f6be6032ea6727d55937587a940aceb76069d2b197890b9`.
Its AIME-2026 gate is running in job `17568`, with automatic paired comparison
job `17569`.

### 8.11 Decisive-core semantic contingency (S16)

S16 is implemented but remains conditional on S15 failing its strict
AIME-2026 gate. It returns to S10's positive-plan barycenter, exponential
retraction, prefix-512 horizon, and six-step optimizer schedule; the only
configuration change is `fisher_plan_core_sentences: 2`. Before rendering,
both guide and matched-control plans are reduced to their first two complete
sentences. Short, answer-claim, and within-ensemble duplicate cores hard fail.

The complete suite passes 323 tests plus 70 subtests. An exact integration
audit over the immutable selected cache validated all 192 examples and all
1,152 guide/control views. Every example retains three distinct guide cores
and three distinct control cores; the selection remains exactly 48 examples
per algebra, geometry, number theory, and combinatorics. Guide cores average
337.4 characters, with range 182--562, matching the preregistered semantic
audit. No S16 training or benchmark evaluation is launched unless S15 fails,
so it remains an isolated contingency rather than a combined method.

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
