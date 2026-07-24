import sys
import unittest
from unittest.mock import patch

from opsd_research.launch_official_opsd import _validate_invocation


MODEL = "Qwen/Qwen3-1.7B"
REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"


def invocation(
    student_revision: str = REVISION, max_steps: str = "200"
) -> list[str]:
    return [
        "launch_official_opsd",
        "--model_name_or_path",
        MODEL,
        "--model_revision",
        REVISION,
        "--student_model_revision",
        student_revision,
        "--max_steps",
        max_steps,
        "--student_thinking",
        "--teacher_thinking",
        "--fixed_teacher",
        "--use_peft",
    ]


class TrainingLaunchTests(unittest.TestCase):
    def test_matching_rollout_revision_is_accepted(self):
        with patch.object(sys, "argv", invocation()):
            _validate_invocation()

    def test_unpinned_rollout_revision_is_rejected(self):
        with patch.object(sys, "argv", invocation("main")):
            with self.assertRaisesRegex(SystemExit, "student_model_revision must match"):
                _validate_invocation()

    def test_one_step_smoke_requires_explicit_environment_gate(self):
        with patch.object(sys, "argv", invocation(max_steps="1")):
            with patch.dict("os.environ", {}, clear=False):
                with self.assertRaisesRegex(SystemExit, "memory smoke"):
                    _validate_invocation()
            with patch.dict(
                "os.environ", {"OPSD_SMOKE_MAX_STEPS": "1"}, clear=False
            ):
                _validate_invocation()


if __name__ == "__main__":
    unittest.main()
