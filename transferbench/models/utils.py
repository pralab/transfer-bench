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


class ImageResizer(nn.Module):
    r"""Resize input image to the specified size."""

    def __init__(self, size: tuple[int, int]) -> None:
        r"""Initialize the ImageResizer module.

        Parameters
        ----------
        - size (tuple[int, int]): Target size for resizing.
        """
        super().__init__()
        self.size = size

    def forward(self, inputs: Tensor) -> Tensor:
        r"""Resize input image to the specified size."""
        if inputs.shape[:2] != self.size:
            return nn.functional.interpolate(inputs, size=self.size)
        return inputs

    def __repr__(self) -> str:
        r"""Return the string representation of the ImageResizer module."""
        return f"ImageResizer(size={self.size})"


def add_resizing(model: Module, size: tuple[int, int]) -> Module:
    r"""Add a resizing layer to the model."""
    return nn.Sequential(ImageResizer(size), model)


def add_normalization(model: Module, mean: tuple[float], std: tuple[float]) -> Module:
    r"""Add a normalization layer to the model."""
    return nn.Sequential(ImageNormalizer(mean, std), model)
