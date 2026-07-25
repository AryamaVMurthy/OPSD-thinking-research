import os
import subprocess
from pathlib import Path


def test_accept_math_eval_has_guarded_acceptance_pipeline() -> None:
    script = Path("infra/local/accept_math_eval.sh")
    contents = script.read_text(encoding="utf-8")

    assert os.access(script, os.X_OK)
    assert '"${state}" != "COMPLETED"' in contents
    assert '"${exit_code}" != "0:0"' in contents
    assert "transfer-manifest-${job_id}.sha256" in contents
    assert 'sha256sum -c "${manifest_name}"' in contents
    assert "-m opsd_research.rescore_matharena" in contents
    assert "-m opsd_research.compare_math" in contents
    assert "-m opsd_research.review_rollouts" in contents
    assert "artifact-manifest-local-official.sha256" in contents

    result = subprocess.run(
        [str(script)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "usage: accept_math_eval.sh" in result.stderr


def test_archive_eval_requires_review_and_reproducible_builds() -> None:
    script = Path("infra/local/archive_eval.sh")
    contents = script.read_text(encoding="utf-8")

    assert os.access(script, os.X_OK)
    assert "artifact-manifest-local-official.sha256" in contents
    assert "${run_name}-manual-notes.md" in contents
    assert 'sha256sum -c "${manifest_name}"' in contents
    assert "--sort=name" in contents
    assert "--mtime=@0" in contents
    assert '"${first_hash}" != "${second_hash}"' in contents
    assert 'gzip -t "${first_archive}"' in contents

    result = subprocess.run(
        [str(script)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "usage: archive_eval.sh" in result.stderr
