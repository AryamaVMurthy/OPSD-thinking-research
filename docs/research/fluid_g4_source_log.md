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

## Evidence policy

- A source is not counted as evidence for Fluid-G4 quality merely because its
  terminology resembles ours.
- AIME 2025/2026 remain locked for final evaluation and are not used to choose
  the divergence or teacher packet.
- Recent preprints are labeled as preprints; peer review is recorded only when
  stated by the primary source.
