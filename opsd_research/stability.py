"""Low-frequency optimizer stability measurements for LoRA training."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass
class TrainableParameterSnapshot:
    """CPU snapshot used to measure the realized trainable-parameter update."""

    previous: dict[str, torch.Tensor]

    @classmethod
    def capture(cls, model) -> "TrainableParameterSnapshot":
        previous = {
            name: parameter.detach().float().cpu().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        if not previous:
            raise ValueError("model has no trainable parameters to monitor")
        return cls(previous)

    def measure(self, model) -> dict[str, float | int | None]:
        current_parameters = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        if set(current_parameters) != set(self.previous):
            raise RuntimeError("trainable parameter set changed during training")
        parameter_square = 0.0
        update_square = 0.0
        trainable_parameters = 0
        next_snapshot: dict[str, torch.Tensor] = {}
        for name, parameter in current_parameters.items():
            current = parameter.detach().float().cpu()
            previous = self.previous[name]
            if current.shape != previous.shape:
                raise RuntimeError(
                    f"trainable parameter shape changed for {name}"
                )
            parameter_square += float(
                current.double().square().sum().item()
            )
            update_square += float(
                (current - previous).double().square().sum().item()
            )
            trainable_parameters += current.numel()
            next_snapshot[name] = current.clone()
        self.previous = next_snapshot
        parameter_norm = math.sqrt(parameter_square)
        update_norm = math.sqrt(update_square)
        return {
            "trainable_parameters": trainable_parameters,
            "parameter_norm": parameter_norm,
            "update_norm": update_norm,
            "update_to_parameter_ratio": (
                update_norm / parameter_norm
                if parameter_norm > 0.0
                else None
            ),
        }
