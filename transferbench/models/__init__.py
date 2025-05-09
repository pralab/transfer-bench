r"""Load model and offer utils for transfer/attack evaluations."""

from . import utils
from .cifar_models import (
    get_model as get_cifar_model,
)
from .cifar_models import (
    list_models as list_cifar_models,
)
from .imagenet_models import (
    get_model as get_imagenet_model,
)
from .imagenet_models import (
    list_models as list_imagenet_models,
)

OPTIONAL_MODELS = [
    "imagenet_resnet50_pubdef",  # imagenet
    "Xu2024MIMIR_Swin-L",  # imagenet
    "Amini2024MeanSparse_Swin-L",  # imagenet
    "Bartoldson2024Adversarial_WRN-94-16",  # cifar10
    "Peng2023Robust",  # cifar10
    "Amini2024MeanSparse_S-WRN-94-16",  # cifar100
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
                "For robust models, please install all the dependencies "
                " with `pip install transferbench[robust]`."
            )
            raise ImportError(msg) from e
    if "cifar" in model_name:
        return get_cifar_model(model_name)
    return get_imagenet_model(model_name)


def list_models() -> list[str]:
    r"""List all available models."""
    return list_imagenet_models() + list_cifar_models() + OPTIONAL_MODELS


__all__ = ["get_model", "list_models", "utils"]
