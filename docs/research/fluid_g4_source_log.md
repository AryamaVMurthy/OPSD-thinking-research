# Fluid-G4 source log

Last verified: 2026-07-27. `references.bib` is the bibliographic source of
truth. Technical claims below are limited to the cited primary paper; our
interpretations and experimental hypotheses are labeled explicitly.

## `zhao2026opsd`

- Status: arXiv preprint, v3 dated 20 March 2026.
- Verified source: <https://arxiv.org/abs/2601.18734>
- Directly supports: OPSD uses student-generated trajectories; one model is a
  privileged teacher and an answer-blind student; training uses dense
  per-token distribution matching.
- Relevance: this is the base algorithm reproduced by this repository.
- Limitation: the paper does not validate our graph routing, fluid evidence
  packet, or FKL/RKL/JS comparison. Those remain new experimental components.

## `zhu2026manyfaces`

- Status: arXiv preprint, v2 dated 24 May 2026.
- Verified source: <https://arxiv.org/abs/2605.11182>
- Directly supports: math OPD is sensitive to teacher and loss choice; reported
  failure modes include teacher/student mismatch on student prefixes, biased
  Top-K RKL optimization, and loss of instance-specific privileged information
  at inference. The paper reports stop-gradient Top-K, RLVR-adapted teachers,
  and SFT-stabilized students as mitigations in its settings.
- Relevance: motivates exact full-vocabulary objectives, teacher-quality
  checks, prefix-recoverability measurements, and a plain-OPSD control.
- Limitation: its negative OPSD results are setting-specific. They do not prove
  that Fluid-G4 must fail, but they make benchmark gains and mechanistic
  ablations mandatory.

## `agarwal2024gkd`

- Status: peer-reviewed, ICLR 2024; arXiv v3.
- Verified source: <https://arxiv.org/abs/2306.13649>
- Directly supports: on-policy distillation trains on student-generated
  sequences to reduce train/inference sequence mismatch and permits generalized
  divergences when the student cannot exactly express the teacher.
- Relevance: supports using the same on-policy prefixes for all divergence
  arms and treating divergence choice as a real optimization ablation.
- Limitation: this is ordinary teacher/student distillation, not
  instance-privileged self-distillation or action-viability routing.

## `wu2025rethinkingkl`

- Status: peer-reviewed, COLING 2025; arXiv v4.
- Verified source: <https://arxiv.org/abs/2404.02657>
- Directly supports: simplistic “FKL is mean-seeking, RKL is mode-seeking”
  language is unreliable for discrete LLM distillation; the two objectives
  share the same exact matched-distribution optimum but differ during finite
  training, with reported early head/tail emphasis differences.
- Relevance: prevents a folklore-based choice between FKL and RKL. We compare
  exact objectives at matched data, seeds, tokens, and optimization steps.
- Limitation: its AKL results are not evidence for our OPSD teacher or GRAF
  auxiliary. An adaptive mixture is a later hypothesis only if the canonical
  arms establish a stable benefit.

## `zhao2026rosd`

- Status: arXiv preprint, v1 dated 27 May 2026.
- Verified source: <https://arxiv.org/abs/2605.28014>
- Directly supports: reflection-conditioned self-teaching, JS token
  divergence, and suffix-only distillation localized by a quoted first error;
  the paper reports that full-response OPSD can overwrite valid prefixes and
  harm out-of-domain performance. Its reported exact-quote match rate stays
  around 0.5, so hard localization itself has substantial coverage failure.
- Relevance: establishes that reflection plus OPSD and JS are not by
  themselves novel, and motivates a non-AIME transfer check and
  position-resolved diagnostics.
- Limitation: ROSD requires a structured exact error quote and, for wrong
  rollouts, an on-policy correct rollout. It does not test uncertain forced
  continuations, posterior-weighted action supervision, or a
  coverage-preserving free-form packet. Its unmatched quotes fall back to
  full-response distillation rather than supplying a fluid soft mask.

## `tan2026ssopd`

- Status: arXiv preprint, v1 dated 17 May 2026.
- Verified source: <https://arxiv.org/abs/2605.17497>
- Directly supports: contrasting correct and wrong samples from one on-policy
  group can provide dense process supervision; the method selects the shortest
  correct and longest wrong responses and applies a prompt-level frontier
  weight. Its stopping-time analysis defines a fast-success posterior
  proportional to the behavior policy times action value and relates the
  frontier coefficient to action-value variance along the trajectory.
- Relevance: overlaps with the multiple-attempt motivation and makes the
  verified-reference and empirical-continuation contributions subject to
  explicit controls. It also shows that Fluid-G4's value-only Boltzmann action
  target is a stronger update than a behavior-prior-regularized posterior,
  motivating a gated mirror-descent target ablation.
- Limitation: SSOPD is verifier-only and depends on a mixed group containing a
  correct attempt. It does not study a free-form audit with a reference
  fallback or local forced-continuation posteriors.

## `he2026sdzero`

- Status: arXiv preprint, v2 dated 11 June 2026.
- Verified source: <https://arxiv.org/abs/2604.12002>
- Directly supports: a self-reviser can condition on an initial attempt and
  binary reward, and its token distributions can be distilled into the
  generator; the paper reports sample-efficiency gains over matched RL
  baselines.
- Relevance: reinforces the value of converting outcome signals into dense
  self-supervision and narrows any Fluid-G4 novelty claim.
- Limitation: binary whole-response revision is not empirical comparison of
  several local continuations with calibrated uncertainty.

## `yang2026ogls`

- Status: arXiv preprint, v2 dated 29 May 2026.
- Verified source: <https://arxiv.org/abs/2605.12400>; the v2 TeX source was
  inspected directly.
- Directly supports: privileged reflection can impose response-template bias
  and suppress useful reconsideration behavior. OGLS-SD averages teacher
  logits induced by verified positive and negative rollout contexts, adds
  their contrast to unprivileged teacher logits, applies full-vocabulary
  steering only to incorrect rollouts, and anchors the last 128 tokens of
  short correct rollouts with SFT.
- Relevance: outcome-gating correct versus incorrect on-policy trajectories
  and positive-tail length regularization are mandatory baselines, not
  Fluid-G4 novelty. Its reported Qwen3-4B Average@8 improves over OPSD on five
  benchmarks, including AIME 2024.
- Limitation: the reported setup uses eight live rollouts per question,
  8,192-token training generation, clipped forward KL, and only the first
  1,024 rollout tokens for distribution matching. It does not test one-forward
  natural-language compression of a cached multi-attempt, verified-reference,
  forced-continuation evidence packet or exact canonical JS.

## `kim2026tsdkd`

- Status: peer-reviewed ICLR 2026 poster; arXiv 2603.13260.
- Verified sources:
  <https://openreview.net/forum?id=2d0c74e15a71b526477e6f43e48929b2167aa2aa>
  and the arXiv TeX source.
- Directly supports: full-response teacher matching can overwhelm a student.
  TSD-KD uses an entropy-gap gate
  \(\sigma((H(S_t)-H(T_t))/\tau)\) for soft token-selective JSD, restricts
  indirect preference supervision to an early cumulative-entropy “opener,”
  and reports that supervising all tokens is worse than focused selection.
- Relevance: entropy-gap token gating, early-position selection, and
  student-top-k teacher reranking are established techniques. Any Fluid-G4
  soft-localization arm must compare against TSD-KD-style gating and cannot
  claim those components as new.
- Limitation: TSD-KD studies a larger-teacher/smaller-student setting rather
  than privileged self-distillation with a shared base model, and it does not
  use verified multi-attempt free-form audits or empirical forced-continuation
  uncertainty.

## Evidence policy

- A source is not counted as evidence for Fluid-G4 quality merely because its
  terminology resembles ours.
- AIME 2025/2026 remain locked for final evaluation and are not used to choose
  the divergence or teacher packet.
- Recent preprints are labeled as preprints; peer review is recorded only when
  stated by the primary source.
