r"""Download and load the pubdef models from Kaggle."""

import shutil
from pathlib import Path

import kagglehub
import torch

from transferbench.utils.cache import get_cache_dir

from .imagenet_models import get_model as get_imagenet_model

MODELS_DIR = get_cache_dir() / "models" / "pubdef"
MODEL_KAGGLE_LINK = "csitawarin/pubdef-defending-against-transfer-attacks"

ALLOWED_MODELS = [
    # "cifar10_wideresnet34-10_pubdef",
    # "cifar100_wideresnet34-10_pubdef",
    "imagenet_resnet50_pubdef",
]


def download_pubdef_models(models_dir: Path) -> None:
    r"""Download the pubdef models from Kaggle in the pubdef_models_path directory."""
    if not models_dir.exists():
        kaggle_path = Path(kagglehub.dataset_download(MODEL_KAGGLE_LINK))
        models_paths = list(kaggle_path.rglob("*_pubdef"))
        [
            shutil.copytree(
                model_path,
                models_dir / model_path.stem,
            )
            for model_path in models_paths
        ]
        shutil.rmtree(kaggle_path)  # remove other unesesary files


def get_models_path(models_dir: Path) -> dict[str, Path]:
    r"""Load the pubdef models."""
    if not models_dir.exists():
        download_pubdef_models(models_dir)
    return {
        model_dir.stem: model_dir / "checkpoint_best.pt"
        for model_dir in models_dir.rglob("*_pubdef")
    }


def list_models() -> list[str]:
    r"""List all available models from pubdef."""
    models_paths_dict = get_models_path(MODELS_DIR)
    all_models = set(models_paths_dict.keys())
    return list(all_models.intersection(set(ALLOWED_MODELS)))


def get_model_weights(model_name: str) -> dict:
    r"""Load the pubdef model weights."""
    models_paths_dict = get_models_path(MODELS_DIR)
    if model_name not in models_paths_dict:
        error_msg = f"Model {model_name} does not exist."
        raise ValueError(error_msg)
    model_path = str(models_paths_dict[model_name])
    raw_weights = torch.load(model_path, map_location="cpu")["state_dict"]
    return {
        k.replace("module.", "").replace("_wrapped_model.", ""): v
        for k, v in raw_weights.items()
    }


def get_model(model_name: str) -> torch.nn.Module:
    r"""Load the pubdef model."""
    model_weights = get_model_weights(model_name)
    dataset, base_model_name = model_name.split("_")[:2]
    if dataset == "imagenet":
        model = get_imagenet_model(model_id=base_model_name)
    model.load_state_dict(model_weights)
    return model
