r"""Module for taking models from torchvision."""

from torch.nn import Module
from torchvision.models import get_model as get_model_
from torchvision.models import get_model_weights
from torchvision.models import list_models as list_models_
from torchvision.transforms._presets import ImageClassification

from .utils import add_normalization

EXCLUDE_MODELS = ["vit_h_14"]  # minium inputs size is higher than 224x224


def is_an_image_classifier(model_id: str) -> bool:
    """
    Check if the model is an image classifier.

    Parameters
    ----------
    - model_id (str): The identifier for the model, e.g., 'resnet18', 'alexnet'.

    Returns
    -------
    - bool: True if the model is an image classifier, False otherwise.
    """
    weights = get_model_weights(model_id).DEFAULT
    return isinstance(weights.transforms(), ImageClassification)


def is_quantized(model_id: str) -> bool:
    """
    Check if the model is quantized.

    Parameters
    ----------
    - model_id (str): The identifier for the model, e.g., 'resnet18', 'alexnet'.

    Returns
    -------
    - bool: True if the model is quantized, False otherwise.
    """
    return "quantized" in model_id.lower()


def list_models() -> list[str]:
    """
    List all available models from torchvision.

    Returns
    -------
    - list[str]: List of model names.
    """
    return [
        model
        for model in list_models_()
        if is_an_image_classifier(model)
        and not is_quantized(model)
        and model not in EXCLUDE_MODELS
    ]


def get_model(model_id: str) -> list[Module]:
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
    model_weight = get_model_weights(model_id).DEFAULT
    if not is_an_image_classifier(model_id):
        msg = f"{model_id} is not an image classifier."
        raise ValueError(msg)
    mean = model_weight.transforms().mean
    std = model_weight.transforms().std
    return add_normalization(get_model_(model_id, weights=model_weight), mean, std)
