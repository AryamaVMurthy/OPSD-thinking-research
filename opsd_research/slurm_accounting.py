"""Record authoritative allocated GPU-hours for one Slurm job."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
from typing import Any


_GPU = re.compile(r"(?:^|,)gres/gpu=(\d+)(?:,|$)")


def parse_sacct(
    payload: str, *, expected_job_id: str
) -> dict[str, Any]:
    """Parse the exact root-job row from a pipe-delimited ``sacct -X`` call."""
    matches = []
    for line in payload.splitlines():
        if not line.strip():
            continue
        fields = line.split("|")
        if len(fields) != 4:
            raise ValueError(f"unexpected sacct row: {line!r}")
        job_id, state, elapsed_raw, allocated = fields
        if job_id == expected_job_id:
            matches.append((state, elapsed_raw, allocated))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one root row for Slurm job {expected_job_id}"
        )
    state, elapsed_raw, allocated = matches[0]
    try:
        elapsed_seconds = int(elapsed_raw)
    except ValueError as error:
        raise ValueError("Slurm ElapsedRaw must be an integer") from error
    gpu_match = _GPU.search(allocated)
    if elapsed_seconds < 0 or gpu_match is None:
        raise ValueError("Slurm row lacks valid elapsed time or GPU allocation")
    gpu_count = int(gpu_match.group(1))
    if gpu_count < 1:
        raise ValueError("Slurm GPU allocation must be positive")
    return {
        "schema_version": 1,
        "slurm_job_id": expected_job_id,
        "state": state,
        "elapsed_seconds": elapsed_seconds,
        "allocated_gpus": gpu_count,
        "allocated_gpu_seconds": elapsed_seconds * gpu_count,
        "allocated_gpu_hours": elapsed_seconds * gpu_count / 3600.0,
        "accounting_basis": "Slurm ElapsedRaw × allocated gres/gpu",
    }


def query_sacct(job_id: str) -> dict[str, Any]:
    result = subprocess.run(
        [
            "sacct",
            "-X",
            "-j",
            job_id,
            "--format=JobIDRaw,State,ElapsedRaw,AllocTRES",
            "--noheader",
            "--parsable2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_sacct(result.stdout, expected_job_id=job_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    accounting = query_sacct(args.job_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(accounting, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(accounting, sort_keys=True))


if __name__ == "__main__":
    main()
