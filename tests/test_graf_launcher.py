import os
from pathlib import Path
import sys
import types
from unittest import mock
import unittest

from opsd_research import launch_graf_opsd
from opsd_research import launch_official_opsd


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

    def test_routed_loss_uses_nonreentrant_gradient_checkpointing(self):
        class FakeModel:
            def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
                self.kwargs = gradient_checkpointing_kwargs
                return "enabled"

        fake_transformers = types.SimpleNamespace(PreTrainedModel=FakeModel)
        with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
            launch_official_opsd._install_nonreentrant_gradient_checkpointing()
            model = FakeModel()
            self.assertEqual(model.gradient_checkpointing_enable(), "enabled")
            self.assertEqual(model.kwargs, {"use_reentrant": False})
            model.gradient_checkpointing_enable({"preserve_rng_state": False})
            self.assertEqual(
                model.kwargs,
                {"preserve_rng_state": False, "use_reentrant": False},
            )
