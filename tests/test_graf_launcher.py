import os
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
from unittest import mock
import unittest

from opsd_research import launch_graf_opsd
from opsd_research import launch_official_opsd


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "reproductions/06_graf_opsd/configs/c0-long-rollout.yaml"


class GrafLauncherTests(unittest.TestCase):
    def test_canonical_token_divergence_is_exported_to_the_loss_hook(self):
        config = {"token_divergence": "js"}
        with mock.patch.dict(os.environ, {}, clear=True):
            launch_graf_opsd._configure_token_divergence(config)
            self.assertEqual(os.environ["OPSD_TOKEN_DIVERGENCE"], "js")

        with mock.patch.dict(
            os.environ, {"OPSD_TOKEN_DIVERGENCE": "reverse_kl"}, clear=True
        ):
            launch_graf_opsd._configure_token_divergence({})
            self.assertNotIn("OPSD_TOKEN_DIVERGENCE", os.environ)

    def test_smoke_is_the_only_short_step_exception(self):
        arguments = [
            "launch_graf_opsd", "--model_name_or_path", "Qwen/Qwen3-4B",
            "--model_revision", "1cfa9a7208912126459214e8b04321603b3df60c",
            "--student_model_revision", "1cfa9a7208912126459214e8b04321603b3df60c",
            "--max_steps", "1", "--max_completion_length", "2048",
            "--student_thinking", "--teacher_thinking", "--fixed_teacher", "--use_peft",
        ]
        environment = {"GRAF_CONFIG": str(CONFIG), "GRAF_SMOKE_MAX_STEPS": "1"}
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

    def test_context_dossier_requires_its_teacher_only_manifest(self):
        arguments = [
            "launch_graf_opsd", "--model_name_or_path", "Qwen/Qwen3-4B",
            "--model_revision", "1cfa9a7208912126459214e8b04321603b3df60c",
            "--student_model_revision", "1cfa9a7208912126459214e8b04321603b3df60c",
            "--max_steps", "25", "--max_completion_length", "2048",
            "--student_thinking", "--teacher_thinking", "--fixed_teacher", "--use_peft",
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "dossiers.jsonl"
            cache.write_text(json.dumps({
                "schema_version": 1, "accepted": True, "example_index": 0,
                "teacher_dossier": "teacher-only", "blind_attempt_count": 3,
            }) + "\n", encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "schema_version": 1, "cache": str(cache),
                "cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
                "requested_examples": 1, "accepted_examples": 1,
                "rejected_examples": 0, "blind_attempts_per_problem": 3,
                "audit_format": "natural_language_v1",
                "schema_based_selection": False,
                "student_answer_context": False, "teacher_reference_context": True,
            }), encoding="utf-8")
            environment = {
                "GRAF_CONFIG": str(
                    ROOT / "reproductions/06_graf_opsd/configs/"
                    "ch0-contrastive-hindsight-2048.yaml"
                ),
                "CH_DOSSIER_MANIFEST": str(manifest),
            }
            with mock.patch.object(
                launch_graf_opsd.sys, "argv", arguments
            ), mock.patch.dict(os.environ, environment, clear=False):
                loaded = launch_graf_opsd._validate_invocation()

        self.assertEqual(loaded["graph_mode"], "context_dossier")
