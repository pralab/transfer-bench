r"""Module for taking models from torchvision."""

import torch
from torch.nn import Module
from torchvision.models import get_model as get_model_

from .utils import Stats, add_normalization


def get_model(
    model_id: str, mean: Stats, std: Stats, device: torch.device = "cuda"
) -> list[Module]:
    """
    Return a model from torchvision based on the model_id.

    The model is normalized with the given mean and std

    Parameters
    ----------
    - model_id (str): The identifier for the model, e.g., 'resnet18', 'alexnet'.
    - mean (tuple[float, float, float]): Mean values for each channel.
    - std (tuple[float, float, float]): Standard deviation values for each channel.

    Returns
    -------
    - torch.nn.Module: The requested model.
    """
    model = add_normalization(get_model_(model_id), mean, std)
    return model.to(device)


def get_availables_models() -> list[str]:
    """Return a list of available models from torchvision."""
    return [
        model_id
        for model_id in get_model_.model_names
        if model_id not in ["mnasnet0_5", "mnasnet1_0"]
    ]


def get_models(
    *model_ids: Module, mean: Stats, std: Stats, device: torch.device
) -> Module:
    r"""Return a list of models from torchvision based on the model_ids."""
    return [get_model(model_id, mean, std, device) for model_id in model_ids]
