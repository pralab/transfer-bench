r"""Robust models from RobustBench."""

from robustbench import load_model
from torch import nn

from transferbench.utils.cache import get_cache_dir

MODELS_CACHE_DIR = get_cache_dir() / "models"
ALLOWED_IMAGENET_MODELS = ["Xu2024MIMIR_Swin-L", "Amini2024MeanSparse_Swin-L"]
ALLOWED_CIFAR10_MODELS = ["Bartoldson2024Adversarial_WRN-94-16", "Peng2023Robust"]
ALLOWED_CIFAR100_MODELS = [
    "Wang2023Better_WRN-70-16",
    "Amini2024MeanSparse_S-WRN-70-16",
]


def get_robustbench_model(
    name: str, dataset: str, threat_model: str = "Linf"
) -> nn.Module:
    r"""Load a model from RobustBench."""
    return load_model(
        model_name=name,
        dataset=dataset,
        threat_model=threat_model,
        model_dir=MODELS_CACHE_DIR,
    )


def list_models() -> list[str]:
    """List all available models from RobustBench."""
    return ALLOWED_IMAGENET_MODELS + ALLOWED_CIFAR10_MODELS + ALLOWED_CIFAR100_MODELS


def get_model(model_name: str) -> nn.Module:
    """Get a model from RobustBench."""
    if model_name in ALLOWED_CIFAR10_MODELS:
        return get_robustbench_model(model_name, dataset="cifar10")
    if model_name in ALLOWED_CIFAR100_MODELS:
        return get_robustbench_model(model_name, dataset="cifar100")
    return get_robustbench_model(model_name, dataset="imagenet")
