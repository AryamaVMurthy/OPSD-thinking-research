from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPENDENCY_HELPER = PROJECT_ROOT / "infra" / "turing" / "job_dependency.sh"


class SlurmDependencyTests(unittest.TestCase):
    def test_completed_successful_job_needs_no_future_dependency(self):
        with tempfile.TemporaryDirectory() as temporary:
            bin_dir = Path(temporary)
            sacct = bin_dir / "sacct"
            sacct.write_text(
                "#!/usr/bin/env bash\n"
                "printf '16051|COMPLETED|0:0\\n'\n",
                encoding="utf-8",
            )
            sacct.chmod(0o755)
            environment = {
                **os.environ,
                "PATH": f"{bin_dir}:{os.environ['PATH']}",
            }

            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    'source "$1"; resolve_afterok_ids 16051',
                    "dependency-test",
                    str(DEPENDENCY_HELPER),
                ],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "")

    def test_active_job_is_retained_in_afterok_id_list(self):
        with tempfile.TemporaryDirectory() as temporary:
            bin_dir = Path(temporary)
            sacct = bin_dir / "sacct"
            sacct.write_text(
                "#!/usr/bin/env bash\n"
                "printf '16057|RUNNING|0:0\\n'\n",
                encoding="utf-8",
            )
            sacct.chmod(0o755)
            environment = {
                **os.environ,
                "PATH": f"{bin_dir}:{os.environ['PATH']}",
            }

            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    'source "$1"; resolve_afterok_ids 16057',
                    "dependency-test",
                    str(DEPENDENCY_HELPER),
                ],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "16057\n")

    def test_failed_job_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            bin_dir = Path(temporary)
            sacct = bin_dir / "sacct"
            sacct.write_text(
                "#!/usr/bin/env bash\n"
                "printf '16052|FAILED|1:0\\n'\n",
                encoding="utf-8",
            )
            sacct.chmod(0o755)
            environment = {
                **os.environ,
                "PATH": f"{bin_dir}:{os.environ['PATH']}",
            }

            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    'source "$1"; resolve_afterok_ids 16052',
                    "dependency-test",
                    str(DEPENDENCY_HELPER),
                ],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("16052", result.stderr)
            self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
