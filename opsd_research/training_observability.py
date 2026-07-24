from __future__ import annotations

from typing import Any


def flush_final_generation_buffer(trainer: Any) -> bool:
    """Persist the last upstream OPSD rollout buffer on the main process."""
    accelerator = trainer.accelerator
    accelerator.wait_for_everyone()
    flushed = False
    if (
        accelerator.is_main_process
        and trainer.state.global_step > 0
        and trainer._generation_outputs_buffer
    ):
        trainer._save_generation_outputs(trainer.state.global_step)
        flushed = True
    accelerator.wait_for_everyone()
    return flushed
