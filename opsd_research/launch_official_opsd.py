from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

from .config import ALLOWED_MODELS
from .trl_compat import configure_structured_dataset_args


PINNED_DATASET = "jasonrqh/Math-CoT-20k"
OFFICIAL_HARDCODED_DATASET = "siyanzhao/Openthoughts_math_30k_opsd"


def _argument(name: str) -> str | None:
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except (ValueError, IndexError):
        return None


def _flag(name: str) -> bool:
    return name in sys.argv


def _validate_invocation() -> None:
    model = _argument("--model_name_or_path")
    revision = _argument("--model_revision")
    student_revision = _argument("--student_model_revision")
    max_steps = _argument("--max_steps")
    if model not in ALLOWED_MODELS:
        raise SystemExit(f"forbidden model {model!r}; expected an instruction Qwen3 checkpoint")
    if model.endswith("-Base"):
        raise SystemExit("Qwen3 Base checkpoints are forbidden")
    if not revision or revision in {"main", "latest"}:
        raise SystemExit("--model_revision must be an immutable commit")
    if student_revision != revision:
        raise SystemExit(
            "--student_model_revision must match --model_revision so vLLM rollouts "
            "use the pinned training checkpoint"
        )
    if max_steps != "200":
        smoke_steps = os.environ.get("OPSD_SMOKE_MAX_STEPS")
        if max_steps != "5" or smoke_steps != max_steps:
            raise SystemExit(
                "--max_steps must be exactly 200, except for the explicit "
                "OPSD_SMOKE_MAX_STEPS=5 memory smoke"
            )
    for flag in ("--student_thinking", "--teacher_thinking", "--fixed_teacher", "--use_peft"):
        if not _flag(flag):
            raise SystemExit(f"required flag is missing: {flag}")


def _install_dataset_redirect() -> None:
    from .training_data import load_math_cot_20k

    revision = os.environ.get("OPSD_DATASET_REVISION")
    if not revision or revision in {"main", "latest"}:
        raise SystemExit("OPSD_DATASET_REVISION must be an immutable commit")

    import datasets

    original_load_dataset = datasets.load_dataset

    def pinned_load_dataset(path, *args, **kwargs):
        if path != OFFICIAL_HARDCODED_DATASET:
            return original_load_dataset(path, *args, **kwargs)
        if args or kwargs:
            raise RuntimeError(
                "official OPSD dataset call unexpectedly supplied arguments; "
                "refusing an ambiguous redirect"
            )
        loaded = load_math_cot_20k(revision)

        def normalize(example):
            return {
                "problem": example["question"],
                "solution": example["response"],
            }

        return loaded.map(
            normalize,
            remove_columns=loaded["train"].column_names,
            desc="Normalizing pinned Math-CoT-20k for official OPSD",
        )

    datasets.load_dataset = pinned_load_dataset


def _install_structured_dataset_compat() -> None:
    """Adapt upstream OPSD's structured collator to current TRL."""
    import opsd_trainer

    original_init = opsd_trainer.OPSDTrainer.__init__

    def compatible_init(self, *args, **kwargs):
        training_args = kwargs.get("args")
        if training_args is None and len(args) >= 2:
            training_args = args[1]
        if training_args is None:
            raise RuntimeError("OPSDTrainer was initialized without training arguments")
        configure_structured_dataset_args(training_args)
        return original_init(self, *args, **kwargs)

    opsd_trainer.OPSDTrainer.__init__ = compatible_init


def _install_exact_jsd_chunking() -> None:
    raw_chunk_size = os.environ.get("OPSD_EXACT_JSD_VOCAB_CHUNK_SIZE")
    if raw_chunk_size is None:
        return
    try:
        chunk_size = int(raw_chunk_size)
    except ValueError as error:
        raise SystemExit(
            "OPSD_EXACT_JSD_VOCAB_CHUNK_SIZE must be a positive integer"
        ) from error
    if chunk_size <= 0:
        raise SystemExit(
            "OPSD_EXACT_JSD_VOCAB_CHUNK_SIZE must be a positive integer"
        )

    import opsd_trainer

    from .jsd import exact_forward_kl_vocab_chunked

    def chunked_loss(
        student_logits,
        teacher_logits,
        labels=None,
        beta=0.5,
        temperature=1.0,
        reduction="batchmean",
        logits_are_probs=False,
        top_k=None,
        token_clip=None,
    ):
        return exact_forward_kl_vocab_chunked(
            student_logits,
            teacher_logits,
            labels,
            beta=beta,
            temperature=temperature,
            reduction=reduction,
            logits_are_probs=logits_are_probs,
            top_k=top_k,
            token_clip=token_clip,
            chunk_size=chunk_size,
        )

    opsd_trainer.OPSDTrainer.generalized_jsd_loss = staticmethod(chunked_loss)
    print(
        '{"event":"exact_jsd_chunking_enabled",'
        f'"vocab_chunk_size":{chunk_size},'
        '"objective":"full_vocab_forward_kl"}',
        flush=True,
    )


def _install_tail_logits_loss() -> None:
    if os.environ.get("OPSD_TAIL_LOGITS_ONLY") is None:
        return
    if os.environ["OPSD_TAIL_LOGITS_ONLY"] != "1":
        raise SystemExit("OPSD_TAIL_LOGITS_ONLY must be exactly 1 when set")

    import opsd_trainer

    from .tail_logits_loss import compute_loss_with_tail_logits

    opsd_trainer.OPSDTrainer.compute_loss = compute_loss_with_tail_logits
    print(
        '{"event":"tail_logits_loss_enabled",'
        '"scope":"generation_tokens_only"}',
        flush=True,
    )


def _install_graf_routed_loss(
    *, routing_targets, branch_loss_weight: float, entropy_floor_fraction: float
) -> None:
    """Install the GRAF branch objective after its cache join is verified."""
    if branch_loss_weight <= 0:
        raise SystemExit("GRAF viability-routed mode requires branch_loss_weight > 0")
    if not 0.0 <= entropy_floor_fraction <= 1.0:
        raise SystemExit("GRAF entropy_floor_fraction must be in [0, 1]")
    import opsd_trainer

    from .tail_logits_loss import compute_loss_with_graf_routing

    opsd_trainer.OPSDTrainer._graf_routing_targets = routing_targets
    opsd_trainer.OPSDTrainer._graf_branch_loss_weight = float(branch_loss_weight)
    opsd_trainer.OPSDTrainer._graf_entropy_floor_fraction = float(entropy_floor_fraction)
    opsd_trainer.OPSDTrainer.compute_loss = compute_loss_with_graf_routing
    print(
        '{"event":"graf_routed_branch_loss_enabled",'
        f'"examples":{len(routing_targets)},'
        f'"branch_loss_weight":{branch_loss_weight},'
        f'"entropy_floor_fraction":{entropy_floor_fraction}' + "}",
        flush=True,
    )


def _install_graf_source_index_collator() -> None:
    """Pass immutable cache indices through the upstream structured collator."""
    from data_collator import SelfDistillationDataCollator
    import torch

    original_call = SelfDistillationDataCollator.__call__

    def graf_call(self, features):
        result = original_call(self, features)
        if not all("graf_source_index" in feature for feature in features):
            raise RuntimeError("GRAF viability routing received a row without graph identity")
        result["graf_source_index"] = torch.tensor(
            [int(feature["graf_source_index"]) for feature in features], dtype=torch.long
        )
        return result

    SelfDistillationDataCollator.__call__ = graf_call
    print('{"event":"graf_source_index_collator_enabled"}', flush=True)


def _install_nonreentrant_gradient_checkpointing() -> None:
    """Make repeated GRAF policy forwards compatible with distributed ZeRO.

    The routed action loss adds a second differentiable policy forward to the
    OPSD forward.  Reentrant activation checkpointing is not supported when a
    distributed rank re-enters the same checkpointed layer in one backward
    graph; DeepSpeed then reports that its partition was reduced twice.
    PyTorch documents non-reentrant checkpointing as the compatible default
    for this pattern.  Preserve explicitly provided kwargs while selecting the
    non-reentrant implementation for GRAF only.
    """
    from transformers import PreTrainedModel

    original = PreTrainedModel.gradient_checkpointing_enable
    if getattr(original, "_graf_nonreentrant_wrapper", False):
        return

    def enable_nonreentrant(self, gradient_checkpointing_kwargs=None):
        kwargs = dict(gradient_checkpointing_kwargs or {})
        kwargs.setdefault("use_reentrant", False)
        return original(self, gradient_checkpointing_kwargs=kwargs)

    enable_nonreentrant._graf_nonreentrant_wrapper = True
    PreTrainedModel.gradient_checkpointing_enable = enable_nonreentrant
    print(
        '{"event":"graf_nonreentrant_gradient_checkpointing_enabled",'
        '"use_reentrant":false}',
        flush=True,
    )


def _install_final_generation_flush() -> None:
    import opsd_trainer

    from .training_observability import flush_final_generation_buffer

    original_train = opsd_trainer.OPSDTrainer.train

    def train_with_final_flush(self, *args, **kwargs):
        result = original_train(self, *args, **kwargs)
        flushed = flush_final_generation_buffer(self)
        if self.accelerator.is_main_process:
            print(
                '{"event":"final_generation_buffer_flush",'
                f'"step":{self.state.global_step},'
                f'"flushed":{str(flushed).lower()}' + "}",
                flush=True,
            )
        return result

    opsd_trainer.OPSDTrainer.train = train_with_final_flush


def main() -> None:
    _validate_invocation()
    _install_dataset_redirect()
    upstream = Path(__file__).resolve().parents[1] / "third_party" / "opsd"
    sys.path.insert(0, str(upstream))
    _install_structured_dataset_compat()
    _install_exact_jsd_chunking()
    _install_tail_logits_loss()
    _install_final_generation_flush()
    runpy.run_path(str(upstream / "opsd_train.py"), run_name="__main__")


if __name__ == "__main__":
    main()
