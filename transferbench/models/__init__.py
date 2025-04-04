r"""Load model and offer utils for transfer/attack evaluations."""

from . import utils
from .imagenet_models import (
    get_model as get_imagenet_model,
)
from .imagenet_models import (
    list_models as list_imagenet_models,
)

OPTIONAL_MODELS = [
    "imagenet_resnet50_pubdef",
    "Xu2024MIMIR_Swin-L",
    "Amini2024MeanSparse_Swin-L",
]


def get_model(model_name: str) -> None:
    r"""Load a model from Imagenet or RobustBench."""
    if model_name in OPTIONAL_MODELS:
        try:
            from .pubdef_models import (
                get_model as get_pubdef_model,
            )
            from .pubdef_models import (
                list_models as list_pubdef_models,
            )
            from .robust_models import (
                get_model as get_robustbench_model,
            )
            from .robust_models import (
                list_models as list_robust_models,
            )

            if model_name in list_pubdef_models():
                return get_pubdef_model(model_name)
            if model_name in list_robust_models():
                return get_robustbench_model(model_name)
        except ImportError as e:
            msg = (
                "For Optional models, please install the full pachage"
                " with `pip install transferbench[full]`."
            )
            raise ImportError(msg) from e

    return get_imagenet_model(model_name)


def list_models() -> list[str]:
    r"""List all available models."""
    return list_imagenet_models() + OPTIONAL_MODELS


__all__ = ["get_model", "list_models", "utils"]
