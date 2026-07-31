# Fisher Self-Information I-Projection Design

## Status and decision rule

This is the pre-registered contingency for S14, not a simultaneous method
change. S14's mixture retraction preserves frozen self-information exactly,
but its six-step training audit shows that the positivity boundary reduces the
requested step at 52.2% of supervised positions. The I-projection below runs
only if the paired S14 AIME-2026 gate fails. It changes only the finite
retraction; guide construction, Fisher tangent, data, optimizer, and evaluation
protocol remain fixed.

## Construction

Let `p` be the frozen next-token distribution, let

\[
h(a)=\log p(a)-\mathbb E_p\log p,
\]

and let `r` be the S13 entropy-neutral consensus score:

\[
\mathbb E_p r=0,\qquad \mathbb E_p[rh]=0.
\]

For a requested tangent scale `alpha`, define

\[
q_{\alpha,\beta}(a)
\propto p(a)\exp\{\alpha r(a)+\beta h(a)\},
\]

where the one-dimensional multiplier `beta` is the unique solution of

\[
\mathbb E_{q_{\alpha,\beta}}h=0.
\]

The moment is monotone in `beta`, with derivative
`Var_q(h)`, so the root is unique for every non-uniform full-support anchor.
Uniform anchors require no correction. The constraint gives

\[
-\mathbb E_q\log p=-\mathbb E_p\log p,
\]

and hence

\[
H(q)=H(p)-D_{\mathrm{KL}}(q\Vert p)\le H(p).
\]

Unlike the S14 mixture geodesic, this exponential-family I-projection has full
support and no positivity step boundary. Because the original direction is
orthogonal to `h`, implicit differentiation at zero gives
`beta'(0)=0`; therefore

\[
\left.\frac{d q}{d\alpha}\right|_{0}=p r,
\]

exactly the same local Fisher tangent as S13 and S14. The worse of forward and
reverse KL is still clipped at 0.01.

## Numerical contract

The implementation uses a bounded Newton moment solve with a monotone bracketed
fallback, float64 reductions for float32 target identities, and the existing
outer KL bisection. It hard-fails on a nonfinite target, an unbracketed or
unconverged moment, a nonpositive probability ratio, a frozen-cross-entropy
residual outside dtype tolerance, entropy increase outside tolerance, or a
two-sided target KL above 0.01.

Before any cluster run, tests must establish:

1. normalization, full support, and exact frozen-cross-entropy preservation;
2. `target_entropy_change = -target_reverse_kl` within dtype tolerance;
3. the same derivative at zero as the original Fisher tangent;
4. no positivity limiting for an extreme negative consensus score;
5. exact two-sided KL clipping;
6. valid configuration and no changes beyond variant/retraction mode; and
7. random float32 stress at the 151,936-token production vocabulary.

## Isolated screen

The S15 configuration differs from S13 only by
`fisher_retraction_mode: self_information_exponential` and its variant name:

- immutable Qwen3-4B revision `1cfa9a7`;
- 192 examples, exactly 48 each from algebra, combinatorics, geometry, and
  number theory;
- three independently generated problem-only plans;
- no answer, reference solution, verifier, reward, correctness, or length
  label;
- 512-token supervised rollout prefix and 32 positions;
- six sequential DP8/GA4 AdamW updates, effective batch 32;
- learning rate `1e-6`, Adam epsilon `1e-4`, linear decay, gradient cap `0.1`;
- LoRA rank 64, alpha 128, anchor KL weight 1, and target KL cap 0.01.

The first task gate is AIME-2026 with six paired rollouts per problem and a
32,768-token cap. AIME-2025 runs only if AIME-2026 preserves average,
majority, pass@6, and cutoff behavior. Promotion requires nonnegative paired
Avg@6 on both years, no majority or pass@6 loss, at most one added cutoff per
year, and exact flip evidence that gains arise from checked reasoning rather
than unsupported commitment.

If this full-support correction still fails, retraction geometry is no longer
the leading hypothesis. The next isolated variable is branch localization:
retain the same target but train only at high-consensus Fisher intervention
positions, rather than uniformly spaced prefix tokens. No further self-
information retraction variant is justified before that test.
