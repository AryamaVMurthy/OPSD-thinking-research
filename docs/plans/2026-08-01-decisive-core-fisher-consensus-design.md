# Decisive-Core Fisher Consensus Design

## Decision status

This is the pre-registered semantic contingency if S15 fails its AIME-2026
gate. It is not combined with S15. The candidate returns to S10, the only
recent clean screen with positive Avg@6 movement on both AIME-2025 and
AIME-2026, and changes only the text span retained from each answer-free plan.

## Evidence

S13--S15 alter the finite Fisher geometry while leaving the plan direction
unchanged. S14 nevertheless loses five net AIME-2026 rollouts, two pass@6
problems, and adds two cutoffs. Its degraded transitions add 4,945 output
tokens on average; improved transitions remove 6,316. Exact degraded paths
frequently abandon checked derivations for a later unsupported route or guess.

The cause is visible in the actual 576 plans used by the equal-domain training
subset. The builder prompt explicitly asks for falsification checks and a
fallback. Of the selected plans, 56.9% mention an alternative, 44.6% mention
verification, 28.8% contain `fallback`, 25.5% contain `critical check`, and
21.2% contain `if this route fails`. In paired S14 AIME-2026 flips, degraded
outputs add 11.5 occurrences of `wait`, 2.5 of `assume`, and 0.94 of `guess`
on average; improved outputs shorten and remove these markers.

The plans follow a consistent order because of the registered generation
prompt: representation first, key deductions second, then checks and fallback.
Keeping exactly the first two sentences is therefore a deterministic semantic
localizer, not a learned selector. On all 576 selected plans it:

- reduces mean plan length from 758 to 337 characters;
- retains at least 182 characters per plan;
- removes every occurrence of `fallback`, `if this route fails`,
  `reconsider`, and `critical check`;
- leaves only 0.3% of cores mentioning verification and 0.2% mentioning an
  alternative; and
- preserves three exactly distinct cores for every one of the 192 problems.

## Method

Let each problem have independent answer-free plans
`g_1, g_2, g_3`. Split a plan only at ordinary sentence-final punctuation and
define its decisive core

\[
c(g_k)=\text{the first two nonempty sentences of }g_k.
\]

Both guide plans and shuffled control plans receive the same transformation
before prompt rendering. No problem, answer, reference solution, rollout,
reward, verifier, correctness flag, or generated token is inspected by the
transformation. Empty, one-sentence, duplicate, or answer-claim cores hard
fail; the registered cache has none.

The Fisher target is then exactly S10:

\[
u_k=(z(c(g_k))-z_0)-\mathbb E_{p_0}[z(c(g_k))-z_0],
\]

\[
a=\frac{\|\bar u\|_{F,p_0}^2}
        {\frac13\sum_k\|u_k\|_{F,p_0}^2},
\qquad r=a\bar u,
\qquad
q\propto p_0\exp(\alpha r),
\]

with the unchanged two-sided 0.01 KL cap and frozen-anchor proximal. Thus the
three plans remain weak procedural learners and Fisher agreement remains the
only boosting rule. The change removes recovery-style evidence before the
weak learners are combined; it adds no extra loss or geometric constraint.

## Isolated screen

S16 differs from S10 only by variant and
`fisher_plan_core_sentences: 2`:

- immutable Qwen3-4B revision `1cfa9a7`;
- the exact same 192 selected examples and 48/domain split;
- the same cached three problem-only independent plans;
- no answer, solution, verifier, reward, correctness, or length label;
- 512-token rollout prefix and 32 uniformly spaced Fisher positions;
- six sequential DP8/GA4 AdamW steps, effective batch 32;
- learning rate `1e-6`, Adam epsilon `1e-4`, linear decay, gradient cap 0.1;
- LoRA rank 64, alpha 128, anchor KL weight 1, and target KL cap 0.01.

Tests must prove exact two-sentence extraction, paired guide/control treatment,
nonempty and distinct cores, unchanged deploy anchor, config isolation, and
answer-free schema. A one-step smoke must show nonzero loss, gradients, target,
and adapter update.

AIME-2026 runs first with six paired rollouts per problem and a 32,768-token
cap. AIME-2025 runs only if average, majority, pass@6, and cutoff behavior are
preserved. Promotion requires nonnegative Avg@6 on both years, no majority or
pass@6 loss, at most one added cutoff per year, and exact flips that retain
checked routes rather than merely shortening outputs.

If S16 fails, plan recovery style is not sufficient to explain the instability.
The next isolated variable is Fisher-position localization within the same
decisive cores; no additional retraction variant is justified.
