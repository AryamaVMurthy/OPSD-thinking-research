import pytest

from opsd_research.slurm_accounting import parse_sacct


def test_parses_exact_root_job_gpu_hours() -> None:
    accounting = parse_sacct(
        "16595|COMPLETED|6000|billing=153,cpu=46,gres/gpu=4,mem=350G,node=1\n"
        "16595.batch|COMPLETED|6000|cpu=46,gres/gpu=4,mem=350G,node=1\n",
        expected_job_id="16595",
    )

    assert accounting["elapsed_seconds"] == 6000
    assert accounting["allocated_gpus"] == 4
    assert accounting["allocated_gpu_seconds"] == 24000
    assert accounting["allocated_gpu_hours"] == pytest.approx(20 / 3)
    assert accounting["accounting_basis"] == (
        "Slurm ElapsedRaw × allocated gres/gpu"
    )


@pytest.mark.parametrize(
    "payload",
    [
        "",
        "16595.batch|COMPLETED|6000|cpu=46,gres/gpu=4\n",
        "16595|COMPLETED|bad|gres/gpu=4\n",
        "16595|COMPLETED|6000|cpu=46\n",
    ],
)
def test_rejects_missing_or_malformed_root_accounting(payload: str) -> None:
    with pytest.raises(ValueError):
        parse_sacct(payload, expected_job_id="16595")
