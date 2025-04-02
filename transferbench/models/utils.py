r"""Utility functions for models."""

import torch
from torch import Tensor, nn
from torch.nn import Module


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


def add_normalization(model: Module, mean: tuple[float], std: tuple[float]) -> Module:
    r"""Add a normalization layer to the model."""
    model_name = model.__class__.__name__
    model = nn.Sequential(ImageNormalizer(mean, std), model)
    model.__class__.__name__ = model_name
    return model
