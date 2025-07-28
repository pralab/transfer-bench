r"""Utility functions for models."""

import torch
from torch import Tensor, nn
from torch.nn import Module
import torch.nn.functional as F
import numpy as np



class ImageNormalizer(nn.Module):
    r"""Normalize input image with mean and std."""

    def __init__(
        self, mean: tuple[float, float, float], std: tuple[float, float, float]
    ) -> None:
        r"""Initialize the ImageNormalizer module.

        Parameters
        ----------
        - mean (tuple[float, float, float]): Mean values for each channel.
        - std (tuple[float, float, float]): Standard deviation values for each channel.
        """
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.as_tensor(std).view(1, 3, 1, 1))

    def forward(self, inputs: Tensor) -> Tensor:
        r"""Normalize input image with mean and std."""
        return (inputs - self.mean) / self.std

    def __repr__(self) -> str:
        r"""Return the string representation of the ImageNormalizer module."""
        return f"ImageNormalizer(mean={self.mean.squeeze()}, std={self.std.squeeze()})"


class ImageResizer(nn.Module):
    r"""Resize input image to the specified size."""

    def __init__(self, size: tuple[int, int]) -> None:
        r"""Initialize the ImageResizer module.

        Parameters
        ----------
        - size (tuple[int, int]): Target size for resizing.
        """
        super().__init__()
        self.size = size

    def forward(self, inputs: Tensor) -> Tensor:
        r"""Resize input image to the specified size."""
        if inputs.shape[2:] != self.size:
            return nn.functional.interpolate(inputs, size=self.size)
        return inputs

    def __repr__(self) -> str:
        r"""Return the string representation of the ImageResizer module."""
        return f"ImageResizer(size={self.size})"


def add_resizing(model: Module, size: tuple[int, int]) -> Module:
    r"""Add a resizing layer to the model."""
    return nn.Sequential(ImageResizer(size), model)


def add_normalization(model: Module, mean: tuple[float], std: tuple[float]) -> Module:
    r"""Add a normalization layer to the model."""
    return nn.Sequential(ImageNormalizer(mean, std), model)



class ThresholdClassifier(nn.Module):
    """
    Wrapper classifier implementing rejection defense based on confidence threshold.
    
    Adds rejection class when max confidence is below threshold. Output has additional
    dimension where last element corresponds to rejection class.
    
    Training mode: Updates threshold to 99th percentile of confidences
    Testing mode: Uses fixed threshold for predictions
    """
    
    def __init__(self, base_model, threshold_eps=0.5, detector_fn=None, percentile=99):
        """
        Initialize ThresholdClassifier.
        
        Args:
            base_model: Base model to wrap
            threshold_eps: Initial confidence threshold for rejection
            detector_fn: Custom confidence function. If None, uses max probability
            percentile: Percentile for threshold update in training mode
        """
        super(ThresholdClassifier, self).__init__()
        self.base_model = base_model
        self.threshold_eps = threshold_eps
        self.detector_fn = detector_fn if detector_fn is not None else self._max_detector
        self.percentile = percentile
        self.confidence_buffer = torch.tensor([], dtype=torch.float32)
        
    def _max_detector(self, probabilities):
        """
        Default detector: returns maximum confidence.
        
        Args:
            probabilities: Class probability tensor
            
        Returns:
            Maximum confidence tensor
        """
        return torch.amax(probabilities, dim=-1)
    
    def forward(self, x):
        """
        Forward pass with rejection capability.
        
        Training mode: Collects confidences for end-of-epoch threshold update
        Testing mode: Uses fixed threshold for predictions
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor with additional dimension for rejection class
        """

        base_logits = self.base_model(x)
        base_probs = F.softmax(base_logits, dim=-1)
        
        confidences = self.detector_fn(base_probs)
        
        if self.training:
            self.confidence_buffer = torch.cat([self.confidence_buffer, confidences.detach().cpu()])
        
        batch_size = base_probs.shape[0]
        num_classes = base_probs.shape[1]
        output_probs = torch.zeros(batch_size, num_classes + 1, device=base_probs.device)
        
        above_threshold = confidences >= self.threshold_eps
        
        output_probs[above_threshold, :-1] = base_probs[above_threshold]
        output_probs[above_threshold, -1] = 0.0  # No rejection
        
        below_threshold = confidences < self.threshold_eps
        output_probs[below_threshold, :-1] = 0.0
        output_probs[below_threshold, -1] = 1.0  #  rejection
        
        return output_probs
    
    def update_threshold(self):
        """
        Update threshold to specified percentile of current epoch confidences.
        Call this at the end of each training epoch.
        """
        if len(self.confidence_buffer) > 0:
            old_threshold = self.threshold_eps
            self.threshold_eps = torch.quantile(self.confidence_buffer, self.percentile / 100.0).item()
            print(f"Threshold updated: {old_threshold:.4f} → {self.threshold_eps:.4f}")
        else:
            print("Warning: No confidences collected, threshold unchanged")
    
    def reset_buffer(self):
        """Reset confidence buffer. Call this at the start of each training epoch."""
        self.confidence_buffer = torch.tensor([], dtype=torch.float32)
    
    def predict(self, x):
        """
        Prediction with rejection. Returns predicted classes where 
        last class (num_classes) indicates rejection.
        
        Args:
            x: Input tensor
            
        Returns:
            Prediction tensor (including rejection class)
        """
        probs = self.forward(x)
        return torch.argmax(probs, dim=-1)
    
    def get_confidences(self, x):
        """
        Calculate confidences for inputs.
        
        Args:
            x: Input tensor
            
        Returns:
            Confidence tensor
        """
        with torch.no_grad():
            base_logits = self.base_model(x)
            base_probs = F.softmax(base_logits, dim=-1)
            return self.detector_fn(base_probs)
    
    def set_threshold(self, new_threshold):
        """
        Manually set rejection threshold.
        
        Args:
            new_threshold: New threshold value
        """
        self.threshold_eps = new_threshold
        
    def get_rejection_rate(self, x):
        """
        Calculate rejection rate for input batch.
        
        Args:
            x: Input tensor
            
        Returns:
            Rejection rate (float)
        """
        confidences = self.get_confidences(x)
        return (confidences < self.threshold_eps).float().mean().item()
