from pathlib import Path


RUNNER = Path("reproductions/06_graf_opsd/run-routed-candidate.sbatch")
FINOD_RUNNER = Path(
    "reproductions/06_graf_opsd/run-finod-candidate.sbatch"
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
