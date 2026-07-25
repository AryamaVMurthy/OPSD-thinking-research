# GRAF-OPSD: isolated Qwen3-4B autoresearch

This directory contains the experimental successor to the archived raw-OPSD
reproduction. It must never modify the historical `reproductions/04_*` run or
reuse its run names.

The active ladder and promotion gates are defined in
[`docs/plans/2026-07-26-graf-opsd-autoresearch.md`](../../docs/plans/2026-07-26-graf-opsd-autoresearch.md).

Artifacts are isolated under `artifacts/graf_opsd/` and use immutable candidate
IDs. A candidate cannot be submitted without a ledger entry, graph-cache
manifest (when applicable), and a validated configuration.
