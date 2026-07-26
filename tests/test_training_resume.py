from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from opsd_research import launch_official_opsd
from opsd_research.training_resume import latest_valid_checkpoint


def _complete(root: Path, step: int) -> Path:
    checkpoint = root / f"checkpoint-{step}"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text("{}", encoding="utf-8")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
    return checkpoint


def test_latest_valid_checkpoint_ignores_partial_and_nonnumeric(tmp_path: Path) -> None:
    assert latest_valid_checkpoint(tmp_path / "missing") is None
    first = _complete(tmp_path, 50)
    _complete(tmp_path, 100)
    partial = tmp_path / "checkpoint-150"
    partial.mkdir()
    (partial / "trainer_state.json").write_text("{}", encoding="utf-8")
    (tmp_path / "checkpoint-latest").mkdir()

    assert latest_valid_checkpoint(tmp_path) == tmp_path / "checkpoint-100"
    assert first.is_dir()


def test_auto_resume_wrapper_injects_latest_checkpoint(tmp_path: Path) -> None:
    expected = _complete(tmp_path, 50)

    class FakeTrainer:
        def train(self, *args, **kwargs):
            self.received = (args, kwargs)
            return "trained"

    fake_module = SimpleNamespace(OPSDTrainer=FakeTrainer)
    with patch.dict("sys.modules", {"opsd_trainer": fake_module}):
        launch_official_opsd._install_auto_resume()
        trainer = FakeTrainer()
        trainer.args = SimpleNamespace(output_dir=str(tmp_path))
        trainer.accelerator = SimpleNamespace(is_main_process=True)
        assert trainer.train() == "trained"

    assert trainer.received == (
        (),
        {"resume_from_checkpoint": str(expected)},
    )
