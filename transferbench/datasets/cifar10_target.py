r"""Dataset for targeted CIFAR10."""

from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100


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
