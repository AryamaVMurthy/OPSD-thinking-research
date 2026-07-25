# Result assembly

Raw generations, execution metadata, checkpoints, and telemetry stay on
node-local scratch while jobs run. Summaries, rollout-review packets, final
LoRA adapters, and immutable run manifests are copied back here after
validation. AIME 2025 and HMMT 2025 form checkpoint curves; AIME 2026 and
LiveCodeBench v6 are evaluated only for untouched and step-200 models.

The consolidated hardware, software, protocol, results, runtime, queue, and
remaining-work snapshot is in
[`current_end_to_end_status.md`](current_end_to_end_status.md).

Accepted untouched results and immutable archive checksums are recorded in
[`untouched_baselines.md`](untouched_baselines.md).

Implementation findings from the live OPSD run are recorded in
[`opsd_training_observations.md`](opsd_training_observations.md).

Accepted post-training results are recorded in
[`opsd_evaluation_progress.md`](opsd_evaluation_progress.md).
