r"""Utility functions for models."""

import torch
from torch import Tensor, nn
from torch.nn import Module

Stats = tuple[float, float, float]


class ImageNormalizer(nn.Module):
    r"""Normalize input image with mean and std."""

    def __init__(
        self, mean: tuple[float, float, float], std: tuple[float, float, float]
    ) -> None:
        r"""Initialize the ImageNormalizer module.

        Parameters
        ----------
        - mean (tuple[float, float, float]): Mean values for each channel.
        - std (tuple[float, float, float]): Standard deviation values for each channel.
        """
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.as_tensor(std).view(1, 3, 1, 1))

    def forward(self, inputs: Tensor) -> Tensor:
        r"""Normalize input image with mean and std."""
        return (inputs - self.mean) / self.std

    def __repr__(self) -> str:
        r"""Return the string representation of the ImageNormalizer module."""
        return f"ImageNormalizer(mean={self.mean.squeeze()}, std={self.std.squeeze()})"


def add_normalization(model: Module, mean: Stats, std: Stats) -> Module:
    r"""Add a normalization layer to the model."""
    model_name = model.__class__.__name__
    model = nn.Sequential(ImageNormalizer(mean, std), model)
    model.__class__.__name__ = model_name
    return model


class ForwardBackwardCounter:
    r"""Hook to count forward and backward passes, considering batch size."""

    def __init__(self, model: Module) -> None:
        r"""Initialize the counter."""
        self.forwarded_batches = 0
        self.backwarded_batches = 0
        self.forwarded_samples = 0
        self.backwarded_samples = 0
        self.attach_(model)

    def forward_hook(self, module: Module, inputs: tuple[Tensor], *args) -> None:
        r"""Count the forward passes."""
        self.forwarded_batches += 1
        self.forwarded_samples += inputs[0].size(0)

    def backward_hook(
        self, module: Module, grad_input: tuple[Tensor], grad_output: tuple[Tensor]
    ) -> None:
        r"""Count backward passes."""
        self.backwarded_batches += 1
        self.backwarded_samples += grad_output[0].size(0)

    def attach_(self, model: Module) -> None:
        r"""Attach hooks to all modules in the model."""
        model.register_forward_hook(self.forward_hook)
        model.register_full_backward_hook(self.backward_hook)

    def reset(self) -> None:
        r"""Reset the counters."""
        self.forwarded_batches = 0
        self.backwarded_batches = 0
        self.forwarded_samples = 0
        self.backwarded_samples = 0

    def __repr__(self) -> str:
        r"""Return the string representation of the counter."""
        return (
            f"Forwarded batches: {self.forwarded_batches}, "
            f"Forwarded samples: {self.forwarded_samples}, "
            f"Backwarded batches: {self.backwarded_batches}, "
            f"Backwarded samples: {self.backwarded_samples}"
        )

    def get_forwards(self) -> int:  # noqa: D102
        return self.forwarded_samples

    def get_backwards(self) -> int:  # noqa: D102
        return self.backwarded_samples


class ModelWrapper(nn.Module):
    r"""Wrapper for a model to count forward and backward passes."""

    def __init__(self, model: Module) -> None:
        r"""Initialize the ModelWrapper module."""
        super().__init__()
        self.counter = ForwardBackwardCounter(model)
        self.model = model
        self.__class__.__name__ = "Wrapped" + model.__class__.__name__

    def forward(self, inputs: Tensor) -> Tensor:
        r"""Forward pass through the model."""
        return self.model(inputs)

    def __repr__(self) -> str:
        r"""Return the string representation of the ModelWrapper module."""
        return f"ModelWrapper({self.model.__class__.__name__}) {self.model.__repr__()}"
