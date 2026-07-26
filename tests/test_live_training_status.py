from pathlib import Path

from opsd_research.live_training_status import render_markdown, summarize_log


def test_summarizes_partial_training_log(tmp_path: Path) -> None:
    log = tmp_path / "train.log"
    log.write_text(
        "\r  2%| | 1/50 [08:09<6:39:26, 489.11s/it]\r"
        "{'loss': 0.0028, 'grad_norm': 0.105459, 'epoch': 0.0}\n"
        "vLLM generation done - elapsed time: 56.77s, prompts: 1, total tokens: 4096, avg length: 4096.0\n"
        "vLLM generation done - elapsed time: 40.0s, prompts: 1, total tokens: 2048, avg length: 2048.0\n",
        encoding="utf-8",
    )
    status = summarize_log(log, 4096)
    assert status["observed_optimizer_steps"] == 1
    assert status["rollouts"]["calls"] == 2
    assert status["rollouts"]["capped_calls"] == 1
    assert status["loss_history"] == [{"loss": 0.0028, "grad_norm": 0.105459}]
    report = render_markdown(status)
    assert "50.0% (1/2)" in report
    assert "0.002800" in report
