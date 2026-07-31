# Fisher Entropy-Neutral Plan Barycenter Design

## Diagnosis

The answer-free plan barycenter has a real Fisher signal, but its task effect
is not preservation-safe.  Prefix length, target scale, proximal strength,
adapter interpolation, and optimizer grouping have each failed as sufficient
repairs.  The failures have two opposite forms:

- full-prefix supervision increased answer diversity and reduced accuracy on
  AIME-2026;
- global-batch supervision increased repeated success on some problems while
  erasing the only successful branch on others.

A categorical-Fisher trust region limits the magnitude of a policy change.
It does not distinguish procedural redistribution from a direction that
globally concentrates or disperses the frozen token distribution.  The next
screen removes that one nuisance component instead of adding another scalar
sweep.

## Alternatives

1. **Frozen-token half-space projection.** Never decrease the probability of
   the token sampled by the frozen policy.  This is strongly conservative but
   biases the method toward pure self-imitation from one sampled trajectory.
2. **Multi-domain Pareto gradient surgery.** Find a common descent direction
   for four domain losses.  This addresses domain interference, but it is a
   substantially more complex optimizer change and does not directly protect
   within-problem rollout coverage.
3. **Entropy-neutral Fisher projection.** Remove only the component of the
   answer-free consensus tangent that changes frozen-policy entropy to first
   order.  This is the recommended screen because it is local, analytic,
   answer-free, and directly addresses both observed signs of route drift.

## Method

Let (p_0) be the frozen deploy distribution and let (r) be the centered
positive-plan consensus direction already constructed by the Fisher plan
barycenter.  Define the centered log-probability direction

\[
h(a)=\log p_0(a)-\mathbb E_{p_0}[\log p_0].
\]

The derivative of entropy along the exponential tilt

\[
p_\eta(a)\propto p_0(a)\exp(\eta r(a))
\]

is

\[
\left.\frac{dH(p_\eta)}{d\eta}\right|_{\eta=0}
=-\langle r,h\rangle_{F,p_0}.
\]

Project the guidance direction onto the Fisher-orthogonal complement of
(h):

\[
r_\perp
=r-
\frac{\langle r,h\rangle_{F,p_0}}
     {\lVert h\rVert_{F,p_0}^2}h.
\]

If the frozen distribution is locally uniform, the denominator is zero and
the entropy derivative is already zero; the projection is therefore the
identity.  Otherwise, (r_\perp) remains centered and satisfies

\[
\langle r_\perp,h\rangle_{F,p_0}=0.
\]

The existing exponential tilt and two-sided KL bisection are then applied to
(r_\perp) without modification.  The objective remains

\[
D_{\mathrm{KL}}(q_\perp\Vert p_\theta)
+D_{\mathrm{KL}}(p_0\Vert p_\theta).
\]

This is first-order entropy preservation, not an entropy reward or a demand
for high uncertainty.  It removes both entropy-increasing and
entropy-decreasing nuisance components while retaining the maximum-energy
procedural direction in the constrained Fisher subspace.

## Exact Screen

Use the empirically safer prefix-512 sequential checkpoint recipe as the
control.  S13 changes only the direction mode:

- 192 examples, exactly 48 each from algebra, combinatorics, geometry, and
  number theory;
- three independently generated problem-only plans;
- no answer, reference solution, verifier, reward, or length label;
- first 512 rollout tokens and 32 selected positions;
- six DP8/GA4 AdamW updates, effective batch 32;
- learning rate (10^{-6}), Adam epsilon (10^{-4}), gradient clip 0.1;
- LoRA rank 64, alpha 128;
- frozen-anchor KL weight 1 and maximum two-sided target KL 0.01.

All later evaluations use only AIME-2025 and AIME-2026, six paired rollouts
per problem, and a 32,768-token generation cap.  AIME-2024 is excluded.

## Diagnostics and Hard Checks

Each Fisher event additionally records:

- entropy-gradient energy;
- Fisher alignment with entropy before and after projection;
- first-order entropy change after projection;
- retained consensus-energy fraction;
- exact finite-step target entropy change.

Hard failures are nonfinite metrics, target KL above 0.01, an uncentered
projected direction, or residual entropy alignment beyond numerical
tolerance on an active entropy direction.

## Tests

1. A synthetic direction containing procedural and entropy components loses
   only its entropy component.
2. An already entropy-neutral direction is unchanged.
3. A uniform frozen distribution is handled without division instability.
4. The projected direction remains Fisher-centered and the target respects
   the two-sided KL cap.
5. Changing shuffled control logits cannot change the positive-plan target.
6. The S13 configuration differs from the prefix-512 control only in its
   registered direction mode.

## Promotion Gate

The screen is promoted only if both allowed AIME years have nonnegative
paired average accuracy, neither majority nor pass-at-six decreases, and
32k cutoffs do not materially increase.  Passing this screen triggers a
larger 512-example run with exactly 128 examples per domain, followed by the
same final two-year protocol.
