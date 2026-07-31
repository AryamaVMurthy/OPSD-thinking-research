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
