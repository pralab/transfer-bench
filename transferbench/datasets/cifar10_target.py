r"""Dataset for targeted CIFAR10."""

import csv
from collections.abc import Callable
from typing import Optional

from torchvision.datasets import CIFAR10


class CIFAR10T(CIFAR10):
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


if __name__ == "__main__":
    cifar10 = CIFAR10T(
        root="/disk2/datasets/CIFAR10", root_target="data/cifar10_targets.csv"
    )
    print(cifar10[1])
    print(cifar10.targets)
