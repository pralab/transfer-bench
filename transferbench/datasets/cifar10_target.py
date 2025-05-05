r"""Dataset for targeted CIFAR10."""

import csv
from collections.abc import Callable
from typing import Optional

from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100


class CIFAR10Tfixed(CIFAR10):
    r"""Dataset for targeted CIFAR10."""

    def __init__(
        self,
        root: str,
        target_file: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        r"""Initialize the dataset.

        Args:
            root (str): Root directory of dataset.
            target_file (str): Path to the target file.
            transform (Callable, optional): A function/transform that takes in an PIL image and returns a transformed version. Defaults to None.
            target_transform (Callable, optional): A function/transform that takes in the target and transforms it. Defaults to None.
            download (bool, optional): If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again. Defaults to False.
        """
        self.target_file = target_file
        all_indeces = self._load_targets()
        self.image_id = all_indeces["image_id"]
        self.tgt_labels = all_indeces["tgt_labels"]
        train = False
        super().__init__(root, train, transform, target_transform, download)

    def _load_targets(self) -> dict:
        r"""Load the target file."""
        image_id = []
        tgt_labels = []
        if self.target_file is not None:
            with open(self.target_file) as csvfile:
                reader = csv.reader(csvfile)
                image_id, tgt_labels = list(zip(*reader, strict=False))
        image_id = list(map(int, image_id[1:]))
        tgt_labels = list(map(int, tgt_labels[1:]))
        return {
            "image_id": image_id,
            "tgt_labels": tgt_labels,
        }

    def __len__(self) -> int:
        r"""Return the length of the dataset."""
        return len(self.image_id)

    def __getitem__(self, idx: int) -> tuple:
        r"""Return the items at the given index."""
        image, gt_label = super().__getitem__(self.image_id[idx])
        tgt_label = self.tgt_labels[idx]
        if self.target_transform is not None:
            gt_label = self.target_transform(gt_label)
            tgt_label = self.target_transform(tgt_label)
        return image, gt_label, tgt_label

    def __repr__(self) -> str:
        r"""Return the string representation of the dataset."""
        return f"{self.__class__.__name__}(root={self.root}, root_target={self.root_target})"


class CIFARTarget(Dataset):
    r"""Base class for targeted CIFAR datasets."""

    @property
    def nclasses(self) -> int:
        r"""Return the number of classes."""
        return len(self.classes)

    def __len__(self) -> int:
        r"""Return the length of the dataset."""
        return 1000

    def __getitem__(self, index: int) -> tuple:
        r"""Return the items at the given index.

        Return
            img (Tensor): The image at the given index.
            label (int): The label at the given index.
            target (int): The target label at the given index.

        The target labels is computed considering the next index in the dataset.
        If the next index is out of bounds, or the next sample has same label,
        the target label is set to the next class modulo nclasses.
        """
        img, label = super().__getitem__(index)

        if index + 1 < len(self):
            new_label = super().__getitem__(index + 1)[1]
        else:
            new_label = (label + 1) % self.nclasses

        target = new_label if new_label != label else (label + 1) % self.nclasses

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, label, target


class CIFAR10T(CIFARTarget, CIFAR10):
    r"""Dataset for targeted CIFAR10."""


class CIFAR100T(CIFARTarget, CIFAR100):
    r"""Dataset for targeted CIFAR100."""
