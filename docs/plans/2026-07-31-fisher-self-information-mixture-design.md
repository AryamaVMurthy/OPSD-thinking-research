# Fisher Self-Information Mixture Retraction Design

## Evidence and diagnosis

S13 projects the answer-free positive-plan barycenter away from the frozen
entropy gradient. The projection is numerically correct: its residual Fisher
alignment is below `3.5e-9`, its target KL is at most 0.01, and 44.8% of the
procedural energy remains. Nevertheless, every one of its 24 training batches
has positive finite target-entropy change, averaging 0.00324. The exponential
map therefore reintroduces at second order what the tangent projection removes
at first order.

The paired task evidence shows two consequences. On AIME-2025, S13 adds eight
cutoffs, concentrated in correct-to-wrong transitions. On AIME-2026, it loses
ten net correct rollouts even though total cutoffs decrease by one. Problems 10
and 26 move from checked base routes to confident invalid alternatives, while
problem 27 improves. A KL ball controls distance but does not control whether
the finite move transfers probability to continuations that were collectively
unlikely under the frozen solver.

The next screen therefore changes the finite retraction only. It preserves the
same guide estimator, same Fisher projection, same local tangent, and same KL
radius, but imposes an exact frozen-self-information conservation law.

## Alternatives considered

1. **Reduce adapter or target scale.** The earlier 3/8 adapter screen reduced
   aggregate drift but still lost pass coverage and added cutoffs. It weakens
   both useful and harmful directions without distinguishing them.
2. **Self-information-corrected exponential map.** Add a scalar temperature
   multiplier to the exponential family and solve a nested moment-matching
   problem at every token. This retains full support but needs an inner root
   solve inside the KL bisection and is substantially more expensive.
3. **Self-information mixture geodesic.** Move affinely in probability space
   along the existing Fisher tangent. This gives the desired moment identity
   analytically, with only a positivity bound and the existing one-dimensional
   KL bisection. This is the selected S14 screen.

If positivity bounds collapse a material fraction of S14 targets, the
correct contingency is alternative 2, not clipping the tangent or adding a
second heuristic.

## Method

Let `p` be the frozen next-token distribution. Let `r` be S13's projected
consensus score, so that

\[
\mathbb E_p[r]=0,
\qquad
\mathbb E_p[r\log p]=0.
\]

The second identity follows from Fisher orthogonality to centered `log p`.
S13 uses the exponential retraction

\[
q^{(e)}_\alpha(a)\propto p(a)\exp(\alpha r(a)).
\]

S14 instead uses the mixture geodesic

\[
q^{(m)}_\alpha(a)=p(a)(1+\alpha r(a)),
\]

with

\[
0\leq\alpha<\min_{r(a)<0}\frac{-1}{r(a)}.
\]

Normalization is exact because `E_p[r]=0`. Its derivative at zero is `p r`,
the same derivative as the exponential map, so the local Fisher guidance is
unchanged. More importantly,

\[
-\sum_a q^{(m)}_\alpha(a)\log p(a)=H(p),
\]

and therefore

\[
H(q^{(m)}_\alpha)
=H(p)-D_{\mathrm{KL}}(q^{(m)}_\alpha\Vert p)
\leq H(p).
\]

Thus the finite target cannot increase entropy or move probability mass, in
aggregate, toward lower-probability frozen tokens. It can still select a bad
route; this is an experimentally testable conservation constraint, not a
correctness guarantee.

Numerically, the implementation uses a dtype-aware safety margin below the
positivity boundary, normalizes once to remove floating-point centering error,
and applies the existing bisection to the worse of forward and reverse KL.
It records positivity limiting, base-cross-entropy residual, and the entropy
identity residual. Any nonfinite target, nonpositive mixture ratio,
normalization failure, self-information residual above tolerance, entropy
increase above tolerance, or target KL above 0.01 is a hard failure.

## Isolated S14 screen

S14 differs from S13 only in `fisher_retraction_mode`:

- Qwen3-4B at immutable revision `1cfa9a7`;
- 192 examples, exactly 48 each from algebra, combinatorics, geometry, and
  number theory;
- three independently generated problem-only plans;
- no answer, solution, verifier, reward, correctness, or length label;
- 512-token rollout prefix and 32 Fisher positions;
- six sequential DP8/GA4 AdamW updates, effective batch 32;
- learning rate `1e-6`, Adam epsilon `1e-4`, linear decay, gradient cap 0.1;
- LoRA rank 64, alpha 128;
- frozen-anchor KL weight 1 and two-sided target-KL cap 0.01.

The training checkpoint advances only if loss and gradients are finite and
nonzero, the target identities hold, the adapter update is nonzero, and
positivity limiting does not collapse the usable target KL or alignment. The
first task screen is AIME-2026 because S13's non-cutoff correctness failure is
the stricter falsification. It uses six paired rollouts per problem and the
32,768-token cap. AIME-2025 runs only if AIME-2026 is nonnegative in average,
majority, and pass@6 with no material cutoff increase.

## Tests

1. The mixture target preserves frozen cross-entropy and satisfies
   `target_entropy_change = -target_reverse_kl` within dtype tolerance.
2. It has the same first derivative as the exponential target at zero.
3. An extreme negative score activates the positivity limit without producing
   a zero or negative ratio.
4. The worse of forward and reverse target KL never exceeds 0.01.
5. Uniform anchors and zero directions remain stable identities.
6. Nonfinite inputs and an incompatible direction/retraction pairing fail.
7. The S14 configuration differs from S13 only by variant and retraction mode.
8. Training aggregation and summaries expose all conservation diagnostics.
9. Random float32 and float64 stress tests cover skewed logits and production
   tensor shapes.

## Promotion and scale-up

Promotion requires nonnegative paired Avg@6 on both AIME-2025 and AIME-2026,
no majority or pass@6 loss on either year, at most one additional cutoff per
year, and qualitative flips that replace unsupported branching with checked
derivations. A passing screen advances to 1,024 training examples selected as
exactly 256 per AIME domain, with provenance and difficulty/length strata
preserved. The larger run then receives a fresh six-rollout, 32k validation on
AIME-2025 and AIME-2026 only.
