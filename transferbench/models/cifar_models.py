r"""Models available from `chenyaofo/pytorch-cifar-models`."""

import torch
from torch import nn
from transformers import AutoImageProcessor, AutoModelForImageClassification

from .utils import add_normalization

REPO_LINK = "chenyaofo/pytorch-cifar-models"

MEAN = (0.49139968, 0.48215841, 0.44653091)
STD = (0.2023, 0.1994, 0.2010)

ADDITIONAL_HUGG_MODELS = {
    "cifar10_swin_b": "Weili/swin-base-patch4-window7-224-in22k-finetuned-cifar10",
    "cifar10_swin_t": "Skafu/swin-tiny-patch4-window7-224-cifar10",
    "cifar10_vit_b16": "aaraki/vit-base-patch16-224-in21k-finetuned-cifar10",
    "cifar10_beit_b16": "jadohu/BEiT-finetuned",
    "cifar10_convnext_t": "ahsanjavid/convnext-tiny-finetuned-cifar10",
    "cifar100_swin_b": "MazenAmria/swin-base-finetuned-cifar100",
    "cifar100_swin_t": "MazenAmria/swin-tiny-finetuned-cifar100",
}


EXCLUDE_CHEN_MODELS = [
    "cifar10_vit_b16",
    "cifar10_vit_b32",
    "cifar10_vit_l16",
    "cifar10_vit_l32",
    "cifar10_vit_h14",
    "cifar100_vit_b16",
    "cifar100_vit_b32",
    "cifar100_vit_l16",
    "cifar100_vit_l32",
    "cifar100_vit_h14",
]


def list_models() -> list[str]:
    """List of available models."""
    chen_models = [
        model
        for model in torch.hub.list(REPO_LINK, force_reload=False)
        if model not in EXCLUDE_CHEN_MODELS
    ]
    return sorted(chen_models + list(ADDITIONAL_HUGG_MODELS.keys()))


def get_chen_model(model: str, pretrained: bool = True) -> nn.Module:
    """Return the neural model."""
    model = torch.hub.load(REPO_LINK, model, pretrained=pretrained)
    if model.__class__.__name__ == "RepVGG":
        model.convert_to_inference_model()
    model.eval()
    return add_normalization(model, MEAN, STD)


class HuggingfaceModelWrapper(nn.Module):
    """Wrapper for Huggingface models."""

    def __init__(self, model: nn.Module, processor: AutoImageProcessor) -> None:
        """Initialize the model."""
        super().__init__()
        self.model = model
        self.processor = processor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        inputs = self.processor(x, do_rescale=False, return_tensors="pt")
        return self.model(**inputs).logits


def get_hugg_model(model: str) -> nn.Module:
    r"""Return a model from huggingface repo."""
    model_path = ADDITIONAL_HUGG_MODELS[model]
    proc_path = model_path
    if "beit" in model:
        proc_path = ADDITIONAL_HUGG_MODELS["cifar10_vit_b16"]  # bug in beit
    hug_model = AutoModelForImageClassification.from_pretrained(model_path)
    hug_processor = AutoImageProcessor.from_pretrained(proc_path, use_fast=True)
    return HuggingfaceModelWrapper(hug_model, hug_processor)


def get_model(model: str) -> nn.Module:
    r"""Return a model from the available models."""
    if model in ADDITIONAL_HUGG_MODELS:  # prority to huggingface models
        return get_hugg_model(model)
    return get_chen_model(model)
