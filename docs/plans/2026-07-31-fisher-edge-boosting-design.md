# Fisher-Edge Boosting Design

## Problem

Answer-free Fisher Consensus Guidance produces valid, nonzero, KL-bounded
targets, but repeated screens show that its learned displacement is nearly
orthogonal to the target on a new problem. Stronger targets reduce the
student-to-target/frozen-anchor ratio from about 2.1 to 1.25, while neither
learning-rate reduction, Adam calibration, nor a frozen-policy proximal makes
the ratio less than one. Agreement among plans for the same problem therefore
does not establish a transferable training direction.

The next method must test cross-problem transfer without answers, solutions,
rewards, verifiers, or an RL objective. It must preserve the exact
equal-domain subset and existing two-sided KL safety bound.

## Alternatives

1. **More proximal or target-scale tuning.** This is simple but cannot create
   cross-problem alignment. Existing jobs already reject it as a complete
   solution.
2. **Proposal/probe optimizer rollback.** Take a candidate update on one
   subset, evaluate it on a disjoint subset, and accept it only if probe loss
   falls. This measures transfer directly, but requires model and optimizer
   rollback plus roughly doubled Fisher forwards at every step.
3. **Fisher-edge boosting.** Estimate a label-free transfer edge before
   backpropagation and weight only examples whose Fisher score agrees with
   other problems. This is the recommended screen because it is one local,
   auditable addition to the existing method.

## Method

For problem \(i\), selected rollout position \(t\), frozen distribution
\(p_{0,it}\), and within-problem consensus score \(r_{it}\), form the
Fisher-whitened vocabulary signature

\[
s_i =
\operatorname{normalize}\left(
\frac{1}{|T_i|}\sum_{t\in T_i}
\sqrt{p_{0,it}}\odot r_{it}
\right).
\]

The vocabulary coordinate system is shared across problems, while
\(\sqrt{p_0}\) makes the Euclidean inner product equal to the local
categorical Fisher metric. The global data-parallel microbatch contains eight
independent problems. For each example, compute the leave-one-out prototype

\[
m_{-i}=\operatorname{normalize}\left(\sum_{j\ne i}s_j\right)
\]

and its boosting edge

\[
e_i = \max\{0,\langle s_i,m_{-i}\rangle-\tau\}.
\]

Leaving out \(i\) prevents the positive self-similarity bias that would make
random signatures appear coherent. The initial screen uses \(\tau=0\), logs
the full signed edge distribution, and normalizes positive weights to global
mean one. If no example has positive edge, the guidance update is exactly
zero for that microbatch; the frozen-policy proximal remains diagnostic but
must not silently become the only training objective.

The target itself remains the strong, two-sided-KL-bounded target from the
latest screen:

- exponential-tilt ceiling \(\eta=1\);
- maximum forward and reverse target KL 0.01;
- frozen-policy proximal weight \(\lambda=1\);
- 32 positions in the first 1,024 rollout tokens.

The per-example optimization objective is

\[
\mathcal L_i =
\tilde e_i\left[
D_{\mathrm{KL}}(q_i\|p_{\theta,i})
+\lambda D_{\mathrm{KL}}(p_{0,i}\|p_{\theta,i})
\right],
\]

where \(\tilde e_i\) is the globally mean-normalized positive edge.

## Data and Distributed Flow

The existing selector supplies exactly 192 problems: 48 each from algebra,
geometry, number theory, and combinatorics/probability, stratified by source
and question-length quartile. Each rank constructs one detached signature.
An all-gather exchanges eight vocabulary signatures, after which every rank
computes the same edge vector and selects its local weight. No answer-bearing
field is loaded or transmitted.

The 151,936-dimensional float32 all-gather is approximately 4.9 MB per
microbatch across eight ranks, small relative to the existing full-vocabulary
teacher forwards.

## Diagnostics and Failure Handling

Every Fisher loss event must additionally log:

- signed leave-one-out edge;
- normalized positive boost weight;
- active-example fraction;
- global mean and maximum positive edge;
- Fisher signature norm before normalization.

Hard failures:

- nonfinite signature or edge;
- inconsistent gathered rank count;
- nonzero boost weight when the signed edge is nonpositive;
- target KL above 0.01.

A fully inactive microbatch is valid and logs zero guidance optimization
loss. Training is rejected if most microbatches are inactive, because that
means the plans contain no common transferable direction.

## Tests

Unit tests cover:

- identical signatures have unit leave-one-out edge;
- mutually cancelling signatures have no positive edge;
- leave-one-out computation removes self-similarity;
- positive weights have global mean one;
- inactive weights create exactly zero guidance gradient;
- target and proximal diagnostics remain unweighted and visible;
- configuration validation requires DP microbatch size of at least three for
  edge boosting.

The runtime screen uses one exact 192-example epoch (six optimizer steps) on
eight GPUs. Promotion requires:

1. finite nonzero active guidance gradients;
2. meaningful but nontrivial active fraction;
3. maximum target KL at or below 0.01;
4. mean target-loss/frozen-anchor ratio below one after step one;
5. positive Fisher alignment gain;
6. no growth in completion caps in the subsequent AIME-2024 gate.

Failure of this screen rejects vocabulary-signature edge as an estimator and
promotes the more expensive proposal/probe rollback design, rather than
continuing scalar tuning.
