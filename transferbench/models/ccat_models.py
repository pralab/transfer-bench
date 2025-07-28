r"""Download and load the CCAT models from MPI datasets."""

import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import torch

from transferbench.utils.cache import get_cache_dir

MODELS_DIR = get_cache_dir() / "models" / "ccat"
MODEL_LINK = "https://datasets.d2.mpi-inf.mpg.de/arxiv2019-ccat/"

ALLOWED_MODELS = [
    "cifar10_stutz2020_ccat",
]


def create_ccat_resnet(num_classes=10):
    """Create a ResNet that matches CCAT structure exactly using AttackBench implementation."""
    try:
        from AttackBench.attack_evaluation.models.original.stutz2020.resnet import ResNet
        model = ResNet(
            N_class=num_classes,
            resolution=(3, 32, 32)
        )
        return model
    except ImportError as e:
        raise ImportError(
            "Could not import ResNet from AttackBench submodule. "
            "Make sure you have added AttackBench as a submodule with:\n"
            "git submodule add https://github.com/attackbench/AttackBench.git AttackBench\n"
            "git submodule update --init --recursive"
        ) from e


def download_ccat_models(models_dir: Path) -> None:
    r"""Download the CCAT models from MPI datasets in the ccat_models_path directory."""
    if not models_dir.exists():
        zip_filename = "cifar10_ccat.zip"
        zip_url = f"{MODEL_LINK}{zip_filename}"
        zip_path = models_dir / zip_filename
        
        models_dir.mkdir(parents=True, exist_ok=True)

        urllib.request.urlretrieve(zip_url, str(zip_path))
        model_subdir = models_dir / "cifar10_stutz2020_ccat"
        model_subdir.mkdir(exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(model_subdir)
        
        extracted_file = model_subdir / "classifier.pth.tar"
        if extracted_file.exists():
            extracted_file.rename(model_subdir / "checkpoint_best.pth.tar")

        zip_path.unlink()


def get_models_path(models_dir: Path) -> dict[str, Path]:
    r"""Load the CCAT models."""
    if not models_dir.exists():
        download_ccat_models(models_dir)
    return {
        model_dir.stem: model_dir / "checkpoint_best.pth.tar"
        for model_dir in models_dir.rglob("*_ccat")
    }


def list_models() -> list[str]:
    r"""List all available models from CCAT."""
    models_paths_dict = get_models_path(MODELS_DIR)
    all_models = set(models_paths_dict.keys())
    return list(all_models.intersection(set(ALLOWED_MODELS)))


def get_model_weights(model_name: str) -> dict:
    r"""Load the CCAT model weights."""
    models_paths_dict = get_models_path(MODELS_DIR)
    if model_name not in models_paths_dict:
        error_msg = f"Model {model_name} does not exist."
        raise ValueError(error_msg)
    model_path = str(models_paths_dict[model_name])
    raw_weights = torch.load(model_path, map_location="cpu", weights_only=False)[
        "model"
    ]
    return raw_weights


def get_model(model_name: str) -> torch.nn.Module:
    r"""Load the CCAT model."""
    model_weights = get_model_weights(model_name)
    
    dataset, base_model_name = model_name.split("_")[:2]
    if dataset == "cifar10":
        model = create_ccat_resnet(num_classes=10)
    
    model.load_state_dict(model_weights)
    return model