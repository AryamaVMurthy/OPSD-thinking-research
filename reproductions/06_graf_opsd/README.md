# GRAF-OPSD: isolated Qwen3-4B autoresearch

This directory contains the experimental successor to the archived raw-OPSD
reproduction. It must never modify the historical `reproductions/04_*` run or
reuse its run names.

The active ladder and promotion gates are defined in
[`docs/plans/2026-07-26-graf-opsd-autoresearch.md`](../../docs/plans/2026-07-26-graf-opsd-autoresearch.md).

Artifacts are isolated under `artifacts/graf_opsd/` and use immutable candidate
IDs. A candidate cannot be submitted without a ledger entry, graph-cache
manifest (when applicable), and a validated configuration.

G3 is pre-registered as the single-field continuation after an unpromoted G2:
it changes the one-sided branch entropy floor from 0.1 to 1.0 while retaining
the exact same frozen graph, forced-continuation targets, model, corpus, and
evaluation protocol.

Forced-continuation viability is versioned as an assistant-side action prefix:
the cache samples after the exact textual action whose likelihood the routed
loss scores. A manifest from any other forcing protocol is rejected.

If G3 does not promote, C1 is the independent, no-graph 4,096-token OPSD
control. Its five-step smoke is mandatory because the longer on-policy rollout
changes the memory envelope; it is only compared after that feasibility gate.
