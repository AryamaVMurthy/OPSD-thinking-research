# GRAF-OPSD autoresearch plan

## Scope and scientific contract

This program evaluates only thinking-enabled `Qwen/Qwen3-4B` with LoRA
adaptation. Training uses the pinned Math-CoT-20k corpus; no AIME item may
enter training, graph construction, or candidate selection. The historical
OPSD reproduction remains immutable.

The autoresearch development signal is AIME 2024. AIME 2025 and AIME 2026 are
locked confirmation evaluations and are not used to select a configuration.
All reported results use the existing official MathArena scorer and the Qwen3
`Average@12` inference protocol. Cheap pilot evaluations are explicitly
labelled development-only and never reported as final results.

## Success gates

1. Every candidate has a canonical, hashed configuration and a hypothesis in
   the ledger before it is run.
2. A candidate must pass leakage, schema, loss-gradient, and five-step Turing
   smoke tests before a 50-step pilot.
3. A 50-step candidate is promoted only if its paired AIME-2024 development
   estimate is at least +3.0 Avg@12 points versus untouched and the paired
   bootstrap lower bound exceeds zero.
4. A promoted candidate is trained for 200 steps and re-evaluated on AIME
   2024 at the official `n=12` protocol.
5. Only a pre-registered winner is evaluated once on locked AIME 2025 and
   AIME 2026. A positive paper claim requires a positive paired confidence
   interval on both locked sets; otherwise it remains a development result.

## Autoresearch ladder

| ID | Method | Single question answered |
|---|---|---|
| C0 | longer-rollout OPSD control | Does removing the known 1,024-token mismatch help without GRAF? |
| G1 | fork-masked OPSD | Is raw teacher supervision at detected forks the damaging component? |
| G2 | viability-routed GRAF-Lite | Does branch viability provide a useful action-level target? |
| G3 | one-sided action entropy | Does a viable-branch entropy floor prevent premature route collapse? |
| G4 | recovery-routed GRAF | Does local graph-derived recovery improve an already positive G3 signal? |

Only one change is allowed between adjacent candidates. RLVR, PMI
purification, separate formatting training, learned action heads, and full
MCTS are deferred until G2 or G3 has a positive development signal.

## GRAF-Lite data path

For a selected cached subset of Math-CoT problems, a graph builder receives a
problem and reference solution and emits answer-masked JSON:

* up to two meaningful forks;
* three canonical action descriptions per fork;
* preconditions, validation test, risk, and recovery action;
* no final answer, exact reference sentence, or answer-identifying numerical
  intermediate.

Every action is tested by forcing it after an ordinary student prefix and
sampling two unprivileged continuations. The mathematical verifier defines
its empirical viability. The cache stores all prompts, seeds, outputs,
verifier labels, sanitisation decisions, and hashes.

## Initial objective

At a graph-aligned fork, the student matches a distribution over canonical
reasoning actions, not the privileged next-token distribution. Away from
forks, it receives deterministic scaffold KL. The initial objective is:

\[
L = L_{\mathrm{scaffold}} + \lambda_B L_{\mathrm{branch}}
    + \lambda_H L_{\mathrm{one\mbox{-}sided\ entropy}}.
\]

The branch distribution uses empirical forced-continuation viability. Low
alignment confidence disables graph loss; it never forces an unrelated route.

## Autoresearch controller

The controller creates candidates from an allow-listed mutation space,
submits only validated candidates, and records a JSONL ledger. It may change
one of: rollout cap, fork threshold, graph budget, viability temperature,
branch-loss coefficient, or entropy-floor coefficient. It cannot change the
model, training corpus, held-out benchmarks, inference protocol, or promotion
threshold. A human-authored or controller-proposed objective-family change is
logged as a new ladder stage rather than silently selected.

## Resource plan

Graph construction is cached and starts on 512 Math-CoT problems. It uses at
most two forks, three actions, and two forced continuations per fork. The
first Turing job is a five-step eight-A100 smoke. A 50-step pilot follows only
after it is accepted. Full 200-step runs are reserved for promoted candidates.

## Resource-aware contrastive-hindsight extension

The G-series screen exposed two constraints that require a separately named
method family rather than another silent GRAF mutation:

1. only 22 unique routed identities reached the G4 pilot;
2. almost every 2,048-token thinking rollout was cut off before finishing.

The CH family preserves ordinary OPSD's free-form student reasoning. For each
training problem, an answer-blind student samples independent natural
solutions. A privileged auditor then receives those attempts plus the trusted
answer and reference solution and writes an unrestricted prose comparison.
The complete prose dossier is supplied only to the fixed teacher. The student
still receives only the original problem. No JSON schema, fixed method
taxonomy, or formatting-based example selection is permitted.

CH0 uses three 2,048-token blind attempts, a 1,536-token natural audit, and
2,048-token on-policy training. CH1 changes only completion allocation: two
4,096-token blind attempts and 4,096-token on-policy training. Its 12-step
pilot has a maximum rollout budget of 1,572,864 tokens, versus 1,638,400 for
CH0's 25-step pilot, so the comparison does not reward CH1 with more training
tokens.

Admission gates are coverage and context gates, not response-shape gates:

- every requested source identity is retained unless generation itself fails;
- accepted coverage must be at least 95%, and a final run targets 100%;
- exact rendered teacher prompts must fit the configured 28,672-token context;
- undersized context is fixed by increasing the budget, never by silently
  dropping long examples;
- a one-step engineering smoke must complete generation, forward/backward,
  optimization, adapter save, and rollout-integrity checks before a pilot.

The paired development screen uses the same four full-context AIME-2024
samples and seeds for untouched and treatment models. It is a method-selection
screen, not a paper result. A selected method is retrained fresh for 200 steps
at effective batch 32 and evaluated at the official 12-sample protocol.

For CH1 publication-scale training, cache the first 6,912 pinned Math-CoT-20k
identities. The fixed content-hash 5% diagnostic partition leaves exactly
6,526 trainable identities, enough for all 6,400 first-pass examples consumed
by 200 optimizer steps without schema-driven reduction or early recycling.
The preferred allocation is eight GPUs with data parallelism 8 and gradient
accumulation 4; the four-GPU fallback uses accumulation 8. Both preserve the
same effective batch, optimizer, seed, and scientific token budget.
