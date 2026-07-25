import os
from pathlib import Path
from unittest import mock
import unittest

from opsd_research import launch_graf_opsd


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "reproductions/06_graf_opsd/configs/c0-long-rollout.yaml"


class GrafLauncherTests(unittest.TestCase):
    def test_smoke_is_the_only_short_step_exception(self):
        arguments = [
            "launch_graf_opsd", "--model_name_or_path", "Qwen/Qwen3-4B",
            "--model_revision", "1cfa9a7208912126459214e8b04321603b3df60c",
            "--student_model_revision", "1cfa9a7208912126459214e8b04321603b3df60c",
            "--max_steps", "5", "--max_completion_length", "2048",
            "--student_thinking", "--teacher_thinking", "--fixed_teacher", "--use_peft",
        ]
        environment = {"GRAF_CONFIG": str(CONFIG), "GRAF_SMOKE_MAX_STEPS": "5"}
        with mock.patch.object(launch_graf_opsd.sys, "argv", arguments), mock.patch.dict(os.environ, environment, clear=False):
            loaded = launch_graf_opsd._validate_invocation()
        self.assertEqual(loaded["variant"], "c0_long_rollout_control")
