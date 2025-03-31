import torch

from . import datasets
from .dataloader import DataLoader


def get_loader(
    dataset: str, batch_size: int = 128, device: torch.device = "cuda", **kwargs
) -> DataLoader:
    r"""Get the dataloader for the given dataset."""
    dataset = getattr(datasets, dataset)(**kwargs)
    return DataLoader(dataset, batch_size=batch_size, device=device, **kwargs)
