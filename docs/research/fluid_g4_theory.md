# Fluid-G4: theoretical model and falsifiable hypotheses

## Learning problem

For problem \(x\), verified evidence \(z\), and a prefix
\(y_{<t}\sim S_\theta(\cdot\mid x)\), define

\[
S_t = S_\theta(\cdot\mid x,y_{<t}),\qquad
T_t = T(\cdot\mid x,z,y_{<t}).
\]

The base loss is

\[
\mathbb E_{x,y\sim S_\theta}\left[
  \sum_t D(T_t,S_t)
\right].
\]

This retains the core on-policy benefit: supervision is placed on states the
student actually visits, rather than only on reference prefixes
[@agarwal2024gkd; @zhao2026opsd]. Fluid-G4 changes the teacher evidence and
adds a sparse action auxiliary; it does not expose \(z\) to the student.

## What each token divergence changes

For fixed teacher and student distributions:

\[
D_\mathrm{FKL}=\mathrm{KL}(T\|S),\quad
D_\mathrm{RKL}=\mathrm{KL}(S\|T),\quad
D_\mathrm{JS}=\tfrac12\mathrm{KL}(T\|M)+
               \tfrac12\mathrm{KL}(S\|M),
\quad M=\tfrac12(T+S).
\]

All three vanish only when \(S=T\), but their finite-step gradients differ.
For student logits \(a\), FKL has the particularly simple gradient
\(\partial D_\mathrm{FKL}/\partial a=S-T\). RKL weights log-ratio error by the
student distribution, so a student-supported token that the teacher assigns
very low probability can create a sharper correction. JS is symmetric and
bounded by \(\log 2\), and its mixture prevents either side from appearing as
an exact zero in the inner ratios. This makes JS a plausible stability arm,
not an assumed quality winner.

The usual mode-covering/mode-seeking slogan is not a sufficient selection rule
for LLM distillation [@wu2025rethinkingkl]. The preregistered comparison
therefore holds cache, prefixes, seed, token budget, action loss, optimizer,
and temperature fixed.

If several privileged contexts \(z\) are possible for the same observable
state, FKL and RKL also imply different compromises:

\[
\arg\min_S\mathbb E_z\mathrm{KL}(T_z\|S)=\mathbb E_z[T_z],
\]

whereas

\[
\arg\min_S\mathbb E_z\mathrm{KL}(S\|T_z)
\propto \exp(\mathbb E_z[\log T_z]).
\]

The first is an arithmetic teacher mixture; the second is a normalized
geometric consensus. This matters because the student cannot observe
instance-specific privileged evidence at inference. Fluid-G4 must demonstrate
that its teacher supplies reusable correction structure, rather than merely
memorizing a different hidden solution for each problem
[@zhu2026manyfaces].

## Action auxiliary

At a natural fork state \(s\), free-form actions \(a_i\) receive continuation
success observations \(k_i\) out of \(n_i\). With a symmetric
\(\mathrm{Beta}(\alpha,\alpha)\) prior,

\[
\hat v_i=\frac{k_i+\alpha}{n_i+2\alpha}.
\]

A temperature-normalized soft target over non-invalid actions supplies a short
action-level forward KL. Its coefficient is proportional to absolute evidence
strength and is divided by the number of fork opportunities, not by the sum
of confidence weights. Thus a low-confidence single fork is genuinely
attenuated.

The current target is value-only:

\[
q_V(a\mid s)\propto \exp(\hat v(s,a)/\tau).
\]

It should not be confused with the fast-success action posterior analyzed by
SSOPD, which includes the behavior-policy prior,
\(q_F(a\mid s)\propto\pi(a\mid s)Q_F^\pi(s,a)\)
[@tan2026ssopd]. Value-only softmax is a stronger policy-improvement step: it
can give substantial mass to an initially unlikely action based on few
trials. Beta smoothing and absolute evidence attenuation reduce this risk but
do not remove it.

The implemented information weight is also not a complete epistemic
confidence measure. It is the KL of the posterior-mean action target from
uniform, normalized by its maximum. Beta smoothing makes small samples less
extreme, and adaptive sampling spends extra trials on overlapping intervals,
but two forks with identical posterior means and different trial counts can
still receive the same weight.

If the action auxiliary is retained at scale, the uncertainty-calibrated
alternative is

\[
w_f =
\frac{\mathrm{KL}(q_V\|U)}{\log K}
\left(1-\frac{H(P(\arg\max_a v_a))}{\log K}\right),
\]

where juxtaposition denotes multiplication. The second factor is ranking
certainty under independent Beta
posteriors: it vanishes when every measured action is equally likely to be
best and approaches one when the ranking is stable. It can be estimated
deterministically at cache-load time and requires no GPU rollout. Structurally
invalid actions remain separately masked rather than being interpreted as
uncertain measured actions.

If the context-only control matches or beats full Fluid-G4, the first
action-target repair is a dynamic trust-region target

\[
q_{\mathrm{MD}}(a\mid s)\propto
\mathrm{stopgrad}[\pi_\theta(a\mid s)]\,
\exp(\eta\hat v(s,a)),
\]

where juxtaposition denotes multiplication, or equivalently
\(\log q_{\mathrm{MD}}=\log\pi_\theta+\eta\hat v-\log Z\).
This is a mirror-descent policy-improvement step. The detached current policy
is already available from action scoring, so it adds no rollout and adapts as
the student changes. It must be compared at the same forks, continuation
budget, and branch-loss weight. It is not added unless the existing action
auxiliary is empirically implicated, because teacher-visible outcome context
may already carry nearly all useful signal.

This is a contextual-bandit-style auxiliary, not proof that an action is
globally correct. Its principal risks are:

- posterior noise when only two continuations are sampled;
- policy staleness as viability was measured under a frozen earlier student;
- proposal bias if the teacher omits the useful action;
- length bias if long but correct actions have lower sequence likelihood;
- distribution shift when a cached fork state differs from the current
  on-policy prefix.

Adaptive sampling allocates extra continuations only to overlapping action
posteriors. A future refresh ablation is justified only if action target
staleness is measured to be large.

### Auxiliary scale and gradient interference

The implemented objective has the form

\[
L(\theta)=L_{\mathrm{token}}(\theta)
 +\lambda\,\frac{1}{|F|}
 \sum_{f\in F}w_f L_{\mathrm{route},f}(\theta),
\]

where \(w_f\) is absolute evidence strength. This makes a weak single fork
weaker than a strong single fork, but it does not guarantee that the auxiliary
gradient is smaller than the token gradient. A very small token KL and a
moderately difficult action comparison can make the branch/base loss ratio
exceed one even when \(\lambda=0.1\).

Loss magnitude is only a proxy for gradient interference. The development
smoke therefore records branch/base magnitude, total pre-clip gradient norm,
clip exceedance, and realized adapter update together. If the fluid run shows
persistent branch dominance, the next registered repair is a detached
auxiliary trust cap

\[
\widetilde L_{\mathrm{route}}
=L_{\mathrm{route}}\min\left(
1,\frac{\rho\,\mathrm{stopgrad}(L_{\mathrm{token}})}
{\mathrm{stopgrad}(L_{\mathrm{route}})+\epsilon}
\right).
\]

This keeps routing auxiliary in scalar scale without backpropagating through
the controller. It is not introduced pre-emptively: a fixed coefficient is
simpler and remains the primary arm unless measured fluid batches demonstrate
the problem. A stronger publication study should additionally estimate cosine
similarity between token and route gradients on a small diagnostic subset;
doing that on every step would add unnecessary backward passes.

### Two timescales: on-policy prefixes and stale evidence

The current pilot is on-policy with respect to the student trajectory, but its
blind attempts, forced-continuation outcomes, and fixed self-teacher are
frozen. This is desirable for a short causal comparison: refreshing evidence
inside one divergence arm would confound the objective ablation. It is not
automatically appropriate for a long run. As the adapter changes,

- cached fork states may become unlikely under the current policy;
- action success rates measured under the initial policy may cease to predict
  current continuation success; and
- a fixed self-teacher may become either a stabilizing anchor or a performance
  ceiling.

A scalable Fluid-G4 algorithm should therefore use two timescales. Student
rollouts and token updates remain frequent. Evidence refresh is selective and
slow. For cached item \(i\), define a refresh priority from three measured
terms: posterior uncertainty in its action ordering, drift between current
and cached action preferences, and current visitation/recovery frequency.
Only the highest-priority fraction receives new forced continuations. Updated
Beta counts retain the policy version of every trial rather than silently
treating nonstationary trials as exchangeable.

This is a later ablation, not part of the active 12-step arm. The minimum
long-run comparison is:

1. fixed teacher and fixed evidence;
2. fixed teacher with selective evidence refresh;
3. periodically synchronized or EMA teacher with the same refresh schedule.

The refresh interval and budget must be matched in generated tokens. A full
cache rebuild every epoch is neither required for fluidity nor
compute-efficient. No refresh experiment is unlocked unless the fixed-cache
method first has a positive task signal or a measured staleness failure.

## Why the evidence packet must be fluid

Two or three independent answer-blind attempts reveal recurring errors and
recoveries. The verified solution anchors correctness. A natural-language
teacher audit can compare attempts without a named-method taxonomy or rigid
response schema. This preserves coverage: every valid source problem receives
base OPSD even when no reliable action fork exists.

Fluidity does not mean absence of controls. Storage has provenance and
checksums; prompts have strict student/teacher separation; generation must fit
the context window; evidence coverage and leakage are measured. Only the
teacher's mathematical analysis remains unrestricted prose.

## Closest contemporary methods and the actual research boundary

Recent methods make it especially important not to claim novelty for
"reflection plus OPSD" by itself. ROSD already conditions a self-teacher on a
reflection, uses JS divergence, and masks distillation before a quoted first
error [@zhao2026rosd]. SSOPD already contrasts multiple on-policy correct and
wrong attempts and turns the contrast into dense supervision
[@tan2026ssopd]. SD-Zero already conditions a reviser on an attempt and its
binary outcome, then distils the reviser back into the generator
[@he2026sdzero].

The narrower Fluid-G4 hypothesis is the combination of:

1. a free-form comparative audit over several answer-blind attempts and a
   verified reference, without requiring a correct on-policy attempt;
2. empirical forced-continuation outcomes, including uncertainty, shown to
   the teacher as natural language;
3. the same outcome evidence converted into a sparse posterior-weighted
   action auxiliary;
4. coverage-preserving base OPSD for every eligible identity; and
5. an exact-divergence and stability protocol that treats FKL, RKL, and JS as
   controlled alternatives.

This boundary is a hypothesis until the context-only and action-only
attribution controls show that empirical continuation evidence contributes
beyond ordinary reflection. If those controls are null, Fluid-G4 should be
reported as an engineering variant of reflective OPSD, not as a new learning
principle.

ROSD also exposes a serious risk in the current primary arm: full-response
distillation can overwrite valid prefixes. Fluid-G4 does not add a rigid
quoted-error schema to the active run because that would change the registered
method and violate the free-form design goal. Instead, the publication ladder
adds an out-of-domain check and measures correctness as a function of prefix
position. Only if full-response Fluid-G4 improves in-domain performance but
harms transfer will a separately registered *soft localization* arm be tried.
That arm should derive a continuous token weight from already available
teacher disagreement or cached recoverability, rather than requiring a
model-generated error tag. It must be compared against both the unmasked
Fluid-G4 arm and ROSD-style hard localization.

## Applicability beyond contest mathematics

The method is not inherently tied to a fixed answer format. It requires:

1. an observable task input available to the student;
2. a verifier, trusted reference, or outcome signal available during training;
3. two or more answer-blind attempts or trajectories;
4. a teacher capable of turning those trajectories and outcomes into local
   corrective guidance.

For code, the outcome can be tests; for tool use, an execution trace and task
success; for formal reasoning, a proof checker. Tasks with subjective targets
or unreliable verifiers need calibrated human or model preference evidence
instead of binary completion success. The invariant remains the same:
privileged evidence conditions the training teacher, never the deployed
student input.

The extra cost is mostly evidence construction: blind attempts, audit, and a
small number of forced continuations. That cache is amortized across optimizer
steps and divergence ablations. Adaptive continuation allocation and sparse
routing make the marginal training step only one short action-scoring forward
on branch-active microbatches, while every identity still receives ordinary
OPSD. Comparisons must report both cache GPU-hours and training GPU-hours;
excluding cache construction would make the efficiency claim misleading.

## Main failure modes and measurements

1. **Irrecoverable student prefixes.** A privileged teacher may be unable to
   give locally coherent next-token guidance after an early algebraic error.
   Measure teacher/student divergence by prefix correctness and recovery
   outcome, not only its global mean.
2. **Hidden-information aggregation.** The student may average incompatible
   instance-specific teachers. Compare full verified context, answer-only,
   free-form audit, and matched plain OPSD.
3. **Teacher distraction.** Long attempts can dilute the verified solution.
   Measure teacher answer consistency and next-action agreement as packet
   components are added.
4. **Auxiliary dominance.** Branch gradients may overwhelm a small token KL.
   Log base loss, branch loss, gradient norms, effective evidence mass, and
   update norms separately.
5. **Numerical cancellation.** Near-equal 151k-way distributions can yield a
   few micro-nats of negative FP32 residue. Report raw negatives and fail only
   on non-finite or materially negative values using an explicit
   working-precision tolerance; never clamp the optimized objective.
6. **Coverage collapse.** Filtering to informative forks repeats a tiny set.
   Log unique identities, repeat counts, sampling Gini, and the fraction that
   receives base versus auxiliary supervision.
7. **Reward hacking by verbosity.** Log rollout length, cap rate, thinking
   closure, answer extraction, correctness per generated token, and wall time.
8. **Valid-prefix overwrite.** Full-response distillation may alter tokens
   before the first substantive mistake. Measure divergence and downstream
   correctness by normalized prefix position and include an out-of-domain
   development set before considering any localization repair.
9. **Evidence and teacher staleness.** A long-running student can leave cached
   continuation outcomes and a fixed teacher behind. Record policy version,
   current/cached action disagreement, revisitation, and recovery drift;
   selectively refresh only where those measurements justify the cost.

The component attribution order is fixed to avoid spending control compute on
a method with no task effect:

1. full Fluid-G4 versus the untouched paired checkpoint;
2. full Fluid-G4 versus a matched free-form dossier-only control, removing
   both empirical continuation prose and action loss;
3. teacher-visible empirical outcomes without action loss versus full
   Fluid-G4, isolating the sparse auxiliary;
4. matched reference-only OPSD, isolating the complete hindsight packet.

Every control keeps the same 124 identities, canonical divergence, optimizer,
rollout seed, and generated-token budget. Only a positive first comparison
unlocks the more expensive attribution ladder.

## Promotion hypotheses

- H1: exact canonical objectives remain finite and materially non-negative;
- H2: at least two objectives complete five steps without a greater than 15%
  unexplained throughput regression;
- H3: the fluid cache retains at least 95% evidence coverage and 100% of
  otherwise eligible identities in the base stream;
- H4: the promoted method improves paired held-out correctness over both the
  initial model and matched plain OPSD without systematic Pass@k regression;
- H5: the improvement survives two training seeds and is accompanied by
  broader identity coverage and better correct trajectories per GPU-hour.

Failure of H3 rejects the implementation. Failure of H4 rejects the method in
its tested form. One positive AIME sample average is not sufficient evidence.

## Observed stability evidence

The fixed-cache comparison is now complete. Exact FKL and RKL each passed one
optimizer update. Ordinary exact JS exhausted a 46,068 MiB card before its
first backward, but the mathematically equivalent analytic
vocabulary-recomputed backward completed. This distinguishes an autograd
retention failure from a failure of the JS objective.

At five matched fixed-cache updates:

- FKL used at most 43,939 MiB, crossed the 0.1 pre-clip norm at 3/5 updates,
  and ended with update/parameter ratio \(9.66\times10^{-5}\);
- recomputed JS used at most 38,165 MiB, crossed at 1/5 updates, and ended at
  \(9.18\times10^{-5}\);
- both had finite losses, zero material divergence negatives, and comparable
  wall time.

The stronger comparison used the actual 124-row fluid stream with identical
rollouts. FKL produced loss 0.2198 and pre-clip norm 0.2955. JS produced loss
0.0324 and norm 0.02539 at the same wall time and approximately 38.1 GiB peak
memory. This is an 11.6-fold gradient-norm reduction, so JS—not its lower raw
loss scale alone—won the preregistered stability gate.

Globally reduced routing telemetry also resolves an apparent conflict. On the
old 22-identity fixed cache, the route/base ratio could exceed one because
every row was route-active and JS has a smaller scalar base. On the real
coverage-preserving stream, posterior smoothing, absolute information weights,
and sparse routing kept the maximum global ratio below 0.01 in the paired
smoke. In the longer run through six updates, active batches averaged about
0.10 and one reached 0.9765, but none exceeded base loss and no optimizer
gradient reached the clipping threshold. The conditional trust cap is
therefore still not activated. Whether this small auxiliary improves task
quality remains an ablation question; empirical outcomes also affect the
teacher through natural-language context.

The reused viability manifest contains exactly two continuations per action;
the adaptive sampler is implemented for a future cache but did not alter this
immutable pilot evidence. A deterministic 10,000-draw Beta posterior audit of
the full 62-graph cache (before the four held-out identities are removed)
found 168 forks with at least two measured actions. Of these, 142 had uniform
posterior-mean targets and already supply no route. The remaining 26 had
median best-action ranking certainty 0.707, with 7/26 below 0.5. Multiplying
ranking certainty into the current information weight would change its median
from 0.0437 to 0.0309 on these forks. Thus explicit uncertainty calibration
would attenuate the auxiliary further, but the audit does not reveal a
catastrophic hidden target: the active-run auxiliary is already sparse and
subordinate. This evidence supports leaving the registered pilot unchanged
and testing uncertainty-calibrated weighting only after component attribution.
