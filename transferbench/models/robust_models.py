r"""Robust models from RobustBench."""

from robustbench import load_model
from torch import nn

from transferbench.utils.cache import get_cache_dir

MODELS_CACHE_DIR = get_cache_dir() / "models"
MODELS = ["Xu2024MIMIR_Swin-L", "Amini2024MeanSparse_Swin-L"]


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
    return MODELS


def get_model(model_name: str) -> nn.Module:
    """Get a model from RobustBench."""
    return get_robustbench_model(model_name, dataset="imagenet")
