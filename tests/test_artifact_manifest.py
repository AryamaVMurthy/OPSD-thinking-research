import subprocess
from pathlib import Path


def test_artifact_manifest_covers_outputs_and_verifies(tmp_path: Path) -> None:
    (tmp_path / "generations.shard0.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "summary.json").write_text("{}\n", encoding="utf-8")
    script = Path("infra/turing/artifact_manifest.sh").resolve()

    subprocess.run(
        [str(script), str(tmp_path), "artifact-manifest-test.sha256"],
        check=True,
    )

    manifest = tmp_path / "artifact-manifest-test.sha256"
    contents = manifest.read_text(encoding="utf-8")
    assert "generations.shard0.jsonl" in contents
    assert "summary.json" in contents
    assert "artifact-manifest-test.sha256" not in contents
    subprocess.run(["sha256sum", "-c", manifest.name], cwd=tmp_path, check=True)


def test_turing_math_jobs_use_isolated_official_scorer_and_final_manifest() -> None:
    common = Path("infra/turing/common.sh").read_text(encoding="utf-8")
    setup = Path("infra/turing/setup_env.sbatch").read_text(encoding="utf-8")
    evaluation = Path("infra/turing/run_eval.sbatch").read_text(encoding="utf-8")
    lcb_score = Path("infra/turing/run_lcb_score.sbatch").read_text(encoding="utf-8")

    assert "SCORE_ENV_DIR=" in common
    assert '"${UV_BIN}" python install 3.12' in setup
    assert '"${SCORE_ENV_DIR}/bin/python"' in setup
    assert 'score_python="${SCORE_ENV_DIR}/bin/python"' in evaluation
    assert '"${score_python}" -m opsd_research.rescore_matharena' in evaluation
    assert 'official-grades.jsonl' in evaluation
    assert 'summarize_math' not in evaluation
    assert 'artifact_manifest.sh' in evaluation
    assert 'artifact_manifest.sh' in lcb_score
