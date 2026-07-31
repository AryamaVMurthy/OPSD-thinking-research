# Fisher Plan Barycenter Design

## Diagnosis

The answer-free guidance plans and matched controls both contain generic
contest-solving procedure: introduce a representation, test an invariant,
check edge cases, and recover from a failed route. Subtracting an unrelated
control therefore removes not only nuisance style but also the transferable
procedural component the student is meant to learn.

The resulting guide-minus-control residual is problem-specific. This explains
all current observations:

- within-problem plans agree and produce nonzero Fisher targets;
- learned residuals are nearly orthogonal on a new problem;
- stronger targets improve signal-to-drift scale but not alignment;
- cross-problem vocabulary-edge weighting does not help, because it acts
  after the shared component has already been cancelled.

## Alternatives

1. **Soft control subtraction.** Use
   \(u=\bar u_{\text{guide}}-\alpha\bar u_{\text{control}}\) with
   \(0<\alpha<1\). This adds an unidentified coefficient and leaves the same
   cancellation failure in weaker form.
2. **Return to the historical G4 scaffold teacher.** G4 improved both AIME
   years, but its older cache has partial answer-seep and only 22 unique
   active problems.
3. **Answer-free Fisher plan barycenter.** Distill the common direction among
   three independently generated, strictly answer-free positive plans.
   Controls remain an experimental placebo and cache audit, not a term in the
   training direction. This is the recommended method.

## Method

For frozen deploy logits \(z_0\) and logits \(z_k^+\) conditioned on
answer-free plan \(k\), define centered categorical-Fisher score directions

\[
u_k = (z_k^+-z_0)
-\mathbb E_{a\sim p_0}[z_k^+(a)-z_0(a)].
\]

The within-problem coherence is unchanged:

\[
A=
\frac{\|\bar u\|_{F,p_0}^2}
{\frac1K\sum_k\|u_k\|_{F,p_0}^2},
\qquad
\bar u=\frac1K\sum_k u_k.
\]

The retained direction is \(r=A\bar u\), and

\[
q(a)\propto p_0(a)\exp(\eta r(a)).
\]

The screen uses \(\eta=1\), with bisection enforcing both
\(D_{\mathrm{KL}}(p_0\|q)\le0.01\) and
\(D_{\mathrm{KL}}(q\|p_0)\le0.01\). The student objective remains

\[
D_{\mathrm{KL}}(q\|p_\theta)
+D_{\mathrm{KL}}(p_0\|p_\theta).
\]

There is no cross-problem edge weighting in this screen. The method tests one
claim only: positive plan agreement contains a transferable procedural
direction that control subtraction destroyed.

## Leakage and Style Boundary

Every plan was generated from the problem text alone and was rejected if it
contained:

- a boxed expression or answer/result claim;
- a derived numerical equality;
- a numeral absent from the problem;
- reference-solution language or prompt injection.

The barycenter can teach procedural wording, but this is now intentional:
problem-solving procedure is the target capability. Three independently
sampled lenses and Fisher agreement suppress plan-specific phrasing. The
frozen-policy proximal and two-sided KL cap prevent an unconstrained style
takeover.

## Implementation

Add one registered direction mode to target construction:

- `matched_control_residual`: historical guide-minus-control behavior;
- `positive_plan_barycenter`: guide-minus-frozen-base behavior.

The control tensors remain loaded for exact data parity but are not consulted
in barycenter target construction. Runtime logs record the mode, and a unit
test proves that changing controls cannot change a barycenter target.

No optimizer, data, prompt, seed, batch, position, or rollout setting changes
relative to the strong-target proximal control.

## Screen and Gate

Run one exact 192-example epoch: 48 problems each from algebra, geometry,
number theory, and combinatorics/probability.

Promotion requires:

1. nonzero loss and active gradients;
2. maximum two-sided target KL at or below 0.01;
3. noncollapsed positive-plan agreement;
4. mean post-initial target-loss/frozen-anchor ratio below one;
5. positive Fisher three-point alignment;
6. no target or metric dependence on control logits.

If the ratio remains above one, answer-free plan-conditioned next-token
distillation itself lacks transfer at this scale. The next step then reuses
Fisher only as a selection/confidence layer around the empirically successful
G4 objective rather than continuing to alter local logit targets.

## Post-screen trust-region interpolation

The six-step checkpoint improved paired AIME-2024 average accuracy from
133/180 to 137/180 and majority accuracy from 23/30 to 25/30. It nevertheless
introduced four additional 32k cutoffs and reduced pass@6 from 26/30 to 25/30.
Exact paired inspection localized the regression: the six correct-to-wrong
flips grew by 17,339 tokens on average and contained all four new cutoffs,
whereas the nine wrong-to-correct flips shortened by 4,840 tokens on average
and removed three cutoffs.

This is the same basin-selection signature seen across earlier methods, not a
uniform length shift. The Fisher three-point diagnostics also predicted that
guidance alignment should dominate frozen-anchor drift below approximately
0.41 of the learned displacement. Raising the proximal coefficient from one
to four did not realize that smaller displacement. The next causal screen
therefore changes only the deployed LoRA scale to \(3/8\):

\[
\theta_{\mathrm{screen}}
=\theta_0+\frac{3}{8}(\theta_{\mathrm{bary}}-\theta_0).
\]

The adapter tensors, training run, prompts, seeds, sampling protocol, and
benchmark stay fixed. Scaling is implemented by changing PEFT
`lora_alpha` from 128 to 48 in a copied adapter while verifying that the
adapter-weight SHA-256 is unchanged. This directly tests the trust-region
prediction without confounding it with a retraining change.

Promote the interpolation only if it retains a positive average or majority
signal while restoring pass@6 and eliminating the material cutoff increase.
If it erases both gains and regressions, reject post-hoc scaling and test the
separate early-prefix hypothesis.

The \(3/8\) screen falsified displacement magnitude as a sufficient
explanation. Average accuracy moved only from 73.89% to 74.44%, majority from
76.67% to 80.00%, pass@6 still fell from 86.67% to 83.33%, and cutoffs still
rose from 10 to 12. The smaller adapter preserved the AIME I problem 13 gain
at 5/6, but introduced different regressions on AIME II problems 2, 5, 7, and
14. Correct-to-wrong flips remained 9,673 tokens longer on average.

The next test therefore leaves adapter scale, target geometry, optimizer, and
data unchanged and restricts Fisher supervision from the first 1,024 rollout
tokens to the first 512. The causal hypothesis is that plan conditioning is
useful while selecting and instantiating an approach, but its later tokenwise
effect teaches continued reconsideration after the route should already be
committed. This is distinct from optimizing for short outputs: inference
retains the full 32k budget, and the loss still contains no answer, reward,
verifier, length penalty, or termination label.

## Evaluation-protocol amendment

At the user's direction, no subsequent candidate selection or final
evaluation uses AIME-2024. The prefix-512 screen and all later candidates are
evaluated only on AIME-2025 and AIME-2026, with six paired rollouts per
problem and a 32,768-token generation cap. The fixed baselines are:

| Benchmark | Average | Majority | Pass@6 | Cutoffs |
|---|---:|---:|---:|---:|
| AIME-2025 | 112/180 (62.22%) | 22/30 | 26/30 | 15/180 |
| AIME-2026 | 123/180 (68.33%) | 22/30 | 26/30 | 14/180 |

The prefix-512 training run completed in job `17428`. Its post-initial
mechanism diagnostics were slightly worse than prefix 1,024:

| Metric | Prefix 1,024 | Prefix 512 |
|---|---:|---:|
| target-loss / frozen-anchor loss | 1.1540 | 1.1575 |
| Fisher alignment gain | 0.000256 | 0.000217 |
| alignment cosine proxy | 0.0477 | 0.0178 |
| student-anchor forward KL | 0.000629 | 0.000640 |
| clipped target fraction | 18.65% | 23.38% |

This does not promote prefix 512 on internal metrics. Its two allowed task
evaluations still run because the hypothesis concerns long-horizon basin
selection, which the 512-token training loss cannot measure directly.

The exact paired AIME-2026 evaluation completed in jobs `17435`/`17436`.
Prefix 512 moved average accuracy from 123/180 to 124/180 (+0.56 pp),
majority accuracy from 22/30 to 24/30 (+6.67 pp), and pass@6 from 26/30 to
25/30 (-3.33 pp). Cutoffs remained 14/180 and mean output length increased
by only 108 tokens. At sample level there were 13 improvements and 12
regressions. The pass loss was entirely problem 9: the only correct baseline
probability-counting rollout became a completed but unsupported guess, not a
length cutoff. Conversely, all six problem-11 grid-optimization rollouts
became correct (2/6 to 6/6). Thus prefix 512 changes route selection but does
not yet pass the two-benchmark promotion gate; AIME-2025 and the full-prefix
control remain necessary.

The paired AIME-2025 evaluation then completed in jobs `17437`/`17438`.
Average accuracy increased from 112/180 to 114/180 (+1.11 pp), but majority
fell from 22/30 to 21/30, pass@6 fell from 26/30 to 24/30, and cutoffs rose
from 15/180 to 19/180. There were 18 improved and 16 degraded samples.
Correct-to-wrong flips grew by 3,681 tokens on average and introduced three
net cutoffs, while wrong-to-correct flips shortened by 4,960 tokens and
removed four. The two lost pass problems were 11 (2/6 to 0/6, including two
new cutoffs) and 14 (1/6 to 0/6). Prefix 512 therefore gives a small positive
average delta on both years, but it fails the preservation gate and is not
ready for larger-subset promotion.

## Global-batch interference screen

The prefix result falsifies supervised horizon as the main cause of the
preservation failures. The remaining training evidence points to optimizer
interference: the 192-example screen is currently applied as six Adam updates
of 32 examples, while the post-initial Fisher alignment cosine is only 0.0477.
Thus a batch-specific update can be nearly orthogonal to the guidance on the
next group of problems even though every individual target passes the KL cap.

The next minimal screen preserves the exact 192 selected examples, prompts,
targets, seed, learning rate, optimizer, LoRA parameterization, and one-pass
sample exposure. It changes the update grouping from six DP8/GA4 steps to one
DP8/GA24 step, giving an effective batch of 192. The candidate therefore
optimizes the arithmetic mean guidance direction before Adam transforms it,
rather than composing six problem-subset-specific Adam transforms.

This screen deliberately keeps the conservative learning rate at
\(10^{-6}\). Consequently it tests whether a coherent aggregate step is
already useful; it does not assert equality of the integrated six-step Adam
displacement. Promotion still requires positive average accuracy on both
AIME-2025 and AIME-2026 without reducing majority accuracy or pass@6 and
without materially increasing 32k cutoffs.
