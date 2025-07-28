r"""Load model and offer utils for transfer/attack evaluations."""

# default models
from .imagenet_models import (
    get_model as get_imagenet_model,
)
from .imagenet_models import (
    list_models as list_imagenet_models,
)

OPTIONAL_ROBUST_MODELS = [
    "imagenet_resnet50_pubdef",  # imagenet
    "Xu2024MIMIR_Swin-L",  # imagenet
    "Amini2024MeanSparse_Swin-L",  # imagenet
    "Bartoldson2024Adversarial_WRN-94-16",  # cifar10
    "Peng2023Robust",  # cifar10
    "Amini2024MeanSparse_S-WRN-94-16",  # cifar100
    "cifar10_stutz2020_ccat" #cifar10
]


def get_model(model_name: str) -> None:
    r"""Load a model from Imagenet or RobustBench."""
    if model_name in list_imagenet_models():
        return get_imagenet_model(model_name)
    if model_name in OPTIONAL_ROBUST_MODELS:
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
            from .ccat_models import (
                get_model as get_ccat_model,
            )
            from .ccat_models import (
                list_models as list_ccat_models,
            )

            if model_name in list_pubdef_models():
                return get_pubdef_model(model_name)
            if model_name in list_robust_models():
                return get_robustbench_model(model_name)
            if model_name in list_ccat_models():
                return get_ccat_model(model_name)
            
        except ImportError as e:
            msg = (
                "For robust models, please install all the dependencies "
                " with `pip install transferbench[robust]`."
            )
            raise ImportError(msg) from e
    if "cifar" in model_name:
        try:
            from .cifar_models import (
                get_model as get_cifar_model,
            )

            return get_cifar_model(model_name)
        except ImportError as e:
            msg = (
                "For cifar models, please install all the dependencies "
                " with `pip install transferbench[cifar]`."
            )
            raise ImportError(msg) from e
    msg = f"Model {model_name} not found."
    raise ValueError(msg)


def list_models() -> list[str]:
    r"""List all available models."""
    models_list = list_imagenet_models()
    try:
        from .pubdef_models import (
            list_models as list_pubdef_models,
        )
        from .robust_models import (
            list_models as list_robust_models,
        )
        from .ccat_models import (
            list_models as list_ccat_models,
        )
        models_list.extend(list_robust_models() + list_pubdef_models() + list_ccat_models())
    except ImportError:
        pass
    try:
        from .cifar_models import (
            list_models as list_cifar_models,
        )

        models_list.extend(list_cifar_models())
    except ImportError:
        pass

    return sorted(models_list)


__all__ = ["get_model", "list_models"]
