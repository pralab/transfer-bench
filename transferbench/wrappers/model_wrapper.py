r"""Model wrapper to count forward and backward passes."""

from typing import Optional

import torch
from torch import Tensor, nn


class SampleWiseCounter:
    r"""Hook to count forward and backward passes, considering batch size."""

    def __init__(self, model: nn.Module) -> None:
        r"""Initialize the counter."""
        self.forwarded_batches = 0
        self.backwarded_batches = 0
        self.forwarded_samples = None
        self.attach_(model)

    def forward_hook(self, module: nn.Module, inputs: tuple[Tensor], *args) -> None:
        r"""Count the forward passes."""
        self.forwarded_batches += 1

    def backward_hook(
        self, module: nn.Module, grad_input: tuple[Tensor], grad_output: tuple[Tensor]
    ) -> None:
        r"""Count backward passes."""
        self.backwarded_batches += 1

    def attach_(self, model: nn.Module) -> None:
        r"""Attach hooks to all modules in the model."""
        model.register_forward_hook(self.forward_hook)
        model.register_full_backward_hook(self.backward_hook)

    def reset(self, inputs: Tensor) -> None:
        r"""Reset the counters."""
        self.forwarded_batches = 0
        self.backwarded_batches = 0
        self.forwarded_samples = torch.zeros(
            inputs.size(0), dtype=torch.int64, device=inputs.device
        )

    def update(self, mask: Optional[Tensor]) -> None:
        r"""Update the counter with the mask."""
        if mask is None:
            self.forwarded_samples += 1
        else:
            self.forwarded_samples[mask] += 1

    def __repr__(self) -> str:
        r"""Return the string representation of the counter."""
        return (
            f"Forwarded batches: {self.forwarded_batches}, "
            f"Forwarded samples: {self.forwarded_samples.sum().item()}, "
            f"Backwarded batches: {self.backwarded_batches}, "
        )

    def get_forwards(self) -> int:  # noqa: D102
        return self.forwarded_batches

    def get_queries(self) -> int:  # noqa: D102
        return self.forwarded_samples

    def get_backwards(self) -> int:  # noqa: D102
        return self.backwarded_batches


class ModelWrapper(nn.Module):
    r"""Wrapper for a model to count forward and backward passes."""

    def __init__(self, model: nn.Module) -> None:
        r"""Initialize the ModelWrapper module."""
        super().__init__()
        self.counter = SampleWiseCounter(model)
        self.model = model
        self.__class__.__name__ = "Wrapped" + model.__class__.__name__

    def forward(self, inputs: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        r"""Forward pass through the model.

        Args:
            inputs: Input tensor.
            mask: Forward maks to count the number of forward passes sample-wise.
                If None, all samples are counted.
        """
        self.counter.update(mask)
        return self.model(inputs)

    def __repr__(self) -> str:
        r"""Return the string representation of the ModelWrapper module."""
        return f"ModelWrapper({self.model.__class__.__name__}) {self.model.__repr__()}"
