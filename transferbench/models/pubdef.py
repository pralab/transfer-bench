# source code adapted from https://github.com/wagner-group/pubdef/blob/main/src/models/imagenet_resnet.py 
# pip install frozendict

import torch
from torchvision import models
from pathlib import Path
import os
import torch.nn as nn

normalize_params_ = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}

temperature_ = 1.0

class Normalize(nn.Module):
    """Normalize images by mean and std."""

    def __init__(self, mean, std, *args, **kwargs) -> None:
        """Initialize Normalize.

        Args:
            mean: Mean of images per-chaneel.
            std: Std of images per-channel.
        """
        _ = args, kwargs  # Unused
        super().__init__()
        if mean is None or std is None:
            self.mean, self.std = None, None
        else:
            self.register_buffer(
                "mean", torch.tensor(mean)[None, :, None, None]
            )
            self.register_buffer("std", torch.tensor(std)[None, :, None, None])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Normalize images."""
        if self.mean is None:
            return images
        return (images - self.mean) / self.std


class Postprocess(nn.Module):
    """Postprocess logits by temperature scaling."""

    def __init__(self, temperature: float = 1.0) -> None:
        """Initialize Postprocess.

        Args:
            temperature: Temperature scaling. Defaults to 1.0.
        """
        super().__init__()
        if temperature <= 0:
            raise ValueError(
                f"Temperature must be positive, got {temperature}!"
            )
        self._temperature: float = temperature

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by temperature."""
        logits = logits / self._temperature
        # Clamp to avoid overflow/underflow
        logits.clamp_(-32, 32)
        return logits


class ResNet50(nn.Module):
    """Wrapper for ImageNet ResNet model."""

    def __init__(self, **kwargs):
        """Initialize ResNet50 model."""
        super().__init__()
        _ = kwargs  # Unused
        self._wrapped_model = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self._wrapped_model(inputs)

def load_pubdef_resnet50(model_dir: str = 'models', dataset: str = 'imagenet', threat_model: str = 'Linf', model_name: str = 'PubDefResnet50'):
    """
    Load a pre-trained PubDef ResNet-50 model from the specified directory.
    
    Parameters:
        model_dir (str): Base directory where models are stored.
        dataset (str): Dataset name (e.g., 'imagenet').
        threat_model (str): Type of adversarial threat model (e.g., 'Linf').
        model_name (str): Name of the model file (default: 'PubDefResnet50').
    
    Returns:
        torch.nn.Module: The loaded PyTorch model.
    """
    model_dir_ = Path(model_dir) / dataset / threat_model
    model_path = model_dir_ / f'{model_name}.pt'

    if not model_dir_.exists():
        raise FileNotFoundError(f"Model directory '{model_dir_}' not found.")
    
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file '{model_path}' not found.")

    try:
        checkpoint_ = torch.load(model_path, map_location=torch.device('cpu'))["state_dict"]
        checkpoint = {k.replace("module.", ""): v for k, v in checkpoint_.items()}
        print(checkpoint.keys())
        print("\n\n\n")
        architecture = ResNet50(num_classes=1000)

        model = nn.Sequential(
            Normalize(**normalize_params_),
            architecture,
            Postprocess(temperature=temperature_),
        )
        model.load_state_dict(checkpoint)
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading model '{model_path}': {e}")
