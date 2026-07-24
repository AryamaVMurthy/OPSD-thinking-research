from __future__ import annotations

import json

import torch
from transformers import AutoTokenizer

from .prompts import math_messages


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-1.7B",
        revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    )
    prompt = tokenizer.apply_chat_template(
        math_messages("Compute 1+1."),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    if "<think>" not in prompt:
        raise RuntimeError("thinking prefix is absent")
    import flash_attn
    import vllm

    print(
        json.dumps(
            {
                "cuda": torch.version.cuda,
                "gpu_count": torch.cuda.device_count(),
                "gpu_name": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "vllm": vllm.__version__,
                "flash_attn": flash_attn.__version__,
                "thinking_prefix": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
