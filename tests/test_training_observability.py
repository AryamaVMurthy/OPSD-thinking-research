from __future__ import annotations

import types
import unittest

from opsd_research.training_observability import flush_final_generation_buffer


class FakeAccelerator:
    is_main_process = True

    def __init__(self):
        self.barriers = 0

    def wait_for_everyone(self):
        self.barriers += 1


class TrainingObservabilityTests(unittest.TestCase):
    def test_final_nonempty_buffer_is_flushed_once(self):
        trainer = types.SimpleNamespace(
            accelerator=FakeAccelerator(),
            state=types.SimpleNamespace(global_step=200),
            _generation_outputs_buffer=[{"completion": "unfinished"}],
        )
        saved_steps = []

        def save(step):
            saved_steps.append(step)
            trainer._generation_outputs_buffer.clear()

        trainer._save_generation_outputs = save

        self.assertTrue(flush_final_generation_buffer(trainer))
        self.assertEqual(saved_steps, [200])
        self.assertEqual(trainer.accelerator.barriers, 2)

    def test_empty_buffer_is_not_rewritten(self):
        trainer = types.SimpleNamespace(
            accelerator=FakeAccelerator(),
            state=types.SimpleNamespace(global_step=200),
            _generation_outputs_buffer=[],
            _save_generation_outputs=lambda step: self.fail("unexpected save"),
        )

        self.assertFalse(flush_final_generation_buffer(trainer))


if __name__ == "__main__":
    unittest.main()
