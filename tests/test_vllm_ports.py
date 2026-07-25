from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PORT_HELPER = PROJECT_ROOT / "infra" / "turing" / "vllm_port.sh"
EVAL_WORKER = PROJECT_ROOT / "infra" / "turing" / "eval_worker.sh"


class VllmPortTests(unittest.TestCase):
    def run_helper(self, job_id: str, task_id: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "bash",
                "-c",
                'source "$1"; vllm_port_for_task "$2" "$3"',
                "vllm-port-test",
                str(PORT_HELPER),
                job_id,
                task_id,
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_eight_shards_get_disjoint_sixteen_port_blocks(self):
        starts = [
            int(self.run_helper("16070", str(task_id)).stdout)
            for task_id in range(8)
        ]

        self.assertEqual(len(starts), len(set(starts)))
        self.assertEqual(starts, [starts[0] + 16 * offset for offset in range(8)])

    def test_non_numeric_slurm_ids_are_rejected(self):
        result = self.run_helper("not-a-job", "0")

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("invalid Slurm job ID", result.stderr)

    def test_task_ids_outside_eight_gpu_allocation_are_rejected(self):
        result = self.run_helper("16070", "8")

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("invalid Slurm task ID", result.stderr)

    def test_eval_worker_exports_its_task_specific_vllm_port(self):
        expected_port = self.run_helper("16070", "3").stdout.strip()
        with tempfile.TemporaryDirectory() as temporary:
            environment_dir = Path(temporary)
            bin_dir = environment_dir / "bin"
            bin_dir.mkdir()
            (bin_dir / "activate").write_text(
                f'export PATH="{bin_dir}:$PATH"\n',
                encoding="utf-8",
            )
            python = bin_dir / "python3"
            python.write_text(
                "#!/usr/bin/env bash\n"
                "printf 'VLLM_PORT=%s\\n' \"${VLLM_PORT:-unset}\"\n",
                encoding="utf-8",
            )
            python.chmod(0o755)
            result = subprocess.run(
                ["bash", str(EVAL_WORKER)],
                check=False,
                capture_output=True,
                text=True,
                env={
                    "PATH": "/usr/bin:/bin",
                    "PROJECT_SOURCE": str(PROJECT_ROOT),
                    "CONFIG": "dummy.yaml",
                    "EVAL_MODULE": "math_eval",
                    "OUTPUT_DIR": str(environment_dir),
                    "METHOD": "test",
                    "CHECKPOINT": "none",
                    "SLURM_JOB_ID": "16070",
                    "SLURM_PROCID": "3",
                    "SLURM_NTASKS": "8",
                    "ENV_DIR": str(environment_dir),
                },
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"VLLM_PORT={expected_port}", result.stdout)


if __name__ == "__main__":
    unittest.main()
