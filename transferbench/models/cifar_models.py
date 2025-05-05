r"""Models available from `chenyaofo/pytorch-cifar-models`."""

import torch
from torch import nn

from .utils import add_normalization

REPO_LINK = "chenyaofo/pytorch-cifar-models"

MEAN = (0.49139968, 0.48215841, 0.44653091)
STD = (0.2023, 0.1994, 0.2010)


def list_models() -> list[str]:
    """List of available models."""
    return torch.hub.list(REPO_LINK, force_reload=False)


def get_model(model: str, pretrained: bool = True) -> nn.Module:
    """Return the neural model."""
    model = torch.hub.load(REPO_LINK, model, pretrained=pretrained)
    if model.__class__.__name__ == "RepVGG":
        model.convert_to_inference_model()
    model.eval()
    return add_normalization(model, MEAN, STD)
