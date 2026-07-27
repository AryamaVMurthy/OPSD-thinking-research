from pathlib import Path
import unittest
from tempfile import TemporaryDirectory

from opsd_research.live_training_status import render_markdown, summarize_log


class LiveTrainingStatusTests(unittest.TestCase):
    def test_summarizes_canonical_divergence_and_stability_diagnostics(self) -> None:
        with TemporaryDirectory() as directory:
            log = Path(directory) / "canonical.log"
            log.write_text(
                '{"event":"canonical_divergence_loss","objective":"js","value":0.12}\n'
                '{"event":"canonical_divergence_loss","objective":"js","value":0.08}\n'
                '{"event":"divergence_diagnostics","objective":"js","token_count":4,'
                '"forward_kl":{"mean":0.3,"min":0.1,"p50":0.2,"p90":0.5,"p99":0.6,"max":0.7,"nonfinite_count":0,"negative_count":0},'
                '"reverse_kl":{"mean":0.4,"min":0.1,"p50":0.3,"p90":0.6,"p99":0.7,"max":0.8,"nonfinite_count":0,"negative_count":0},'
                '"js":{"mean":0.1,"min":0.02,"p50":0.08,"p90":0.2,"p99":0.25,"max":0.3,"nonfinite_count":0,"negative_count":0},'
                '"student_entropy":{"mean":2.0},"teacher_entropy":{"mean":1.8}}\n'
                '{"event":"adapter_stability","step":1,'
                '"trainable_parameters":100,"parameter_norm":4.0,'
                '"update_norm":0.02,"update_to_parameter_ratio":0.005}\n',
                encoding="utf-8",
            )
            status = summarize_log(log, 4096)

        canonical = status["canonical_divergence"]
        self.assertEqual(canonical["objective"], "js")
        self.assertEqual(canonical["loss_calls"], 2)
        self.assertAlmostEqual(canonical["mean"], 0.1)
        self.assertEqual(canonical["diagnostic_calls"], 1)
        self.assertEqual(canonical["latest"]["js"]["p99"], 0.25)
        self.assertEqual(status["adapter_stability"]["events"], 1)
        self.assertEqual(
            status["adapter_stability"]["latest"]["update_norm"], 0.02
        )
        self.assertIn("Canonical js telemetry", render_markdown(status))
        self.assertIn("Adapter update norm", render_markdown(status))

    def test_summarizes_partial_training_log(self) -> None:
        with TemporaryDirectory() as directory:
            log = Path(directory) / "train.log"
            log.write_text(
                "Normalizing pinned Math-CoT-20k: 100%| | 19428/19428 [00:01<00:00, 13000 examples/s]\n"
                "\r  2%| | 1/50 [08:09<6:39:26, 489.11s/it]\r"
                "{'loss': 0.0028, 'grad_norm': 0.105459, 'epoch': 0.0}\n"
                '{"event":"exact_forward_kl_loss","value":1.25e-06}\n'
                '{"event":"exact_forward_kl_loss","value":2.5e-06}\n'
                '{"event":"graf_branch_loss","active_forks":2.0,"effective_fork_weight":0.5,"branch_kl":0.2,"entropy_floor":0.01,"weighted_loss":0.021}\n'
                '{"event":"graf_branch_loss","active_forks":0.0,"effective_fork_weight":0.0,"branch_kl":0.0,"entropy_floor":0.0,"weighted_loss":0.0}\n'
                "vLLM generation done - elapsed time: 56.77s, prompts: 1, total tokens: 4096, avg length: 4096.0\n"
                "vLLM generation done - elapsed time: 40.0s, prompts: 1, total tokens: 2048, avg length: 2048.0\n",
                encoding="utf-8",
            )
            status = summarize_log(log, 4096)
        self.assertEqual(status["observed_optimizer_steps"], 1)
        self.assertEqual(status["rollouts"]["calls"], 2)
        self.assertEqual(status["rollouts"]["capped_calls"], 1)
        self.assertEqual(status["loss_history"], [{"loss": 0.0028, "grad_norm": 0.105459}])
        self.assertEqual(status["graf_branch"]["active_loss_calls"], 1)
        self.assertEqual(status["graf_branch"]["loss_calls"], 2)
        self.assertEqual(status["graf_branch"]["mean_active_forks_per_loss_call"], 1.0)
        self.assertEqual(status["forward_kl"]["loss_calls"], 2)
        self.assertEqual(status["forward_kl"]["negative_loss_calls"], 0)
        self.assertEqual(status["forward_kl"]["min"], 1.25e-06)
        report = render_markdown(status)
        self.assertIn("`50.0%` (1/2)", report)
        self.assertIn("0.002800", report)
        self.assertIn("Branch-active loss calls: `50.0%` (1/2)", report)
        self.assertIn("Negative loss calls: `0`", report)
