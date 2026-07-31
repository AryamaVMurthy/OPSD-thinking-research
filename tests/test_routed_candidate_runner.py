from pathlib import Path


RUNNER = Path("reproductions/06_graf_opsd/run-routed-candidate.sbatch")
FINOD_RUNNER = Path(
    "reproductions/06_graf_opsd/run-finod-candidate.sbatch"
)
FINOD_PAIRED_SUITE = Path(
    "reproductions/06_graf_opsd/submit-finod-paired-suite.sbatch"
)


def test_runner_selects_manifests_after_reading_graph_mode() -> None:
    script = RUNNER.read_text(encoding="utf-8")

    graph_mode_offset = script.index('graph_mode="$(python3')
    unconditional_graph_offset = script.find(
        ': "${GRAF_GRAPH_CACHE_MANIFEST:?}" "${GRAF_VIABILITY_MANIFEST:?}"'
    )

    assert unconditional_graph_offset == -1
    assert graph_mode_offset < script.index(
        'graph_modes_requiring_graph=('
    )
    assert '"context_dossier"' in script
    assert 'manifest_inputs=(runtime-overrides.txt)' in script


def test_runner_archives_only_the_manifests_used_by_the_mode() -> None:
    script = RUNNER.read_text(encoding="utf-8")

    assert 'if [[ "${requires_graph}" == 1 ]]; then' in script
    assert 'if [[ "${requires_viability}" == 1 ]]; then' in script
    assert 'if [[ "${requires_dossier}" == 1 ]]; then' in script
    assert 'manifest_inputs+=(graph-cache-manifest.json)' in script
    assert 'manifest_inputs+=(viability-manifest.json)' in script
    assert 'manifest_inputs+=(ch-dossier-manifest.json)' in script


def test_runner_uses_the_registered_config_seed() -> None:
    script = RUNNER.read_text(encoding="utf-8")

    assert 'seed="$(python3 - "${GRAF_CANDIDATE_CONFIG}"' in script
    assert '--seed "${seed}" --data_seed "${seed}"' in script
    assert "--seed 42" not in script


def test_runner_archives_fisher_manifest_and_exact_selection() -> None:
    script = RUNNER.read_text(encoding="utf-8")

    assert 'requires_fisher=0' in script
    assert '[[ "${graph_mode}" == "fisher_consensus" ]] && requires_fisher=1' in script
    assert ': "${FISHER_GUIDANCE_MANIFEST:?}"' in script
    assert 'fisher-guidance-manifest.json' in script
    assert 'OPSD_FISHER_SELECTION_MANIFEST' in script
    assert 'fisher-representative-selection.json' in script


def test_runner_summarizes_the_actual_slurm_output_path() -> None:
    script = RUNNER.read_text(encoding="utf-8")

    assert 'training_log="${SLURM_SUBMIT_DIR}/logs/' in script
    assert '--training-log "${training_log}"' in script


def test_guidance_runners_use_the_registered_learning_rate() -> None:
    for path in (RUNNER, FINOD_RUNNER):
        script = path.read_text(encoding="utf-8")
        assert 'learning_rate="$(python3 -' in script
        assert '--learning_rate "${learning_rate}"' in script
        assert "--learning_rate 5e-6" not in script


def test_guidance_runners_use_the_registered_adam_epsilon() -> None:
    for path in (RUNNER, FINOD_RUNNER):
        script = path.read_text(encoding="utf-8")
        assert 'adam_epsilon="$(python3 -' in script
        assert '--adam_epsilon "${adam_epsilon}"' in script


def test_finod_runner_uses_the_registered_update_grouping() -> None:
    script = FINOD_RUNNER.read_text(encoding="utf-8")

    assert 'gradient_accumulation_steps="$(python3 - "${FINOD_CONFIG}"' in script
    assert 'effective_batch_size="$(python3 - "${FINOD_CONFIG}"' in script
    assert script.count(
        '--gradient_accumulation_steps "${gradient_accumulation_steps}"'
    ) == 2
    assert 'gradient_accumulation_steps=%s\\neffective_batch_size=%s' in script
    assert '"${gradient_accumulation_steps}" "${effective_batch_size}"' in script
    assert "gradient_accumulation_steps=4" not in script
    assert "effective_batch_size=32" not in script


def test_finod_paired_suite_is_limited_to_aime25_and_aime26() -> None:
    script = FINOD_PAIRED_SUITE.read_text(encoding="utf-8")

    assert '${FINOD_BENCHMARKS:-aime25 aime26}' in script
    assert "aime25|aime26)" in script
    assert "aime24" not in script
