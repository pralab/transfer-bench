"""Python module to load proper dataset."""

from typing import Optional

from torch.utils.data import Dataset
from torchvision import transforms as tsfm
from torchvision.transforms import InterpolationMode as Interp

from transferbench.utils.cache import get_cache_dir

from .cifar10_target import CIFAR10T as CIFAR10T_
from .cifar10_target import CIFAR100T as CIFAR100T_
from .imagenet_target import ImageNetT as ImageNetT_

DATASET_CACHE_DIR = get_cache_dir() / "datasets"


class BaseDataset(Dataset):
    r"""Abstract class for datasets."""

    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    classes: list[str]
    root: str


class CIFARTntarget(BaseDataset):
    r"""Dataset from `https://www.cs.toronto.edu/~kriz/cifar.html` with targets."""

    mean = (0.49139968, 0.48215841, 0.44653091)
    std = (0.2023, 0.1994, 0.2010)

    def __init__(
        self,
        center: bool = False,
        size: Optional[int] = None,
        target_file: Optional[str] = None,
    ) -> None:
        """Initialize the dataset with target labels."""
        transf_list = []
        if center:
            transf_list.append(tsfm.Normalize(self.mean, self.std))

        if size is not None:
            transf_list.append(tsfm.Resize(size, Interp.NEAREST, antialias=None))

        transf_list.append(tsfm.ToTensor())

        transform = tsfm.Compose(transf_list)
        super().__init__(
            root=self.root,
            train=False,
            transform=transform,
            download=True,
        )


class CIFAR10T(CIFARTntarget, CIFAR10T_):
    r"""Targeted CIFAR10 dataset."""

    root = DATASET_CACHE_DIR / "CIFAR10"


class CIFAR100T(CIFARTntarget, CIFAR100T_):
    r"""Targeted CIFAR100 dataset."""

    root = DATASET_CACHE_DIR / "CIFAR100"


class ImageNetT(ImageNetT_):
    r"""Imagenet target attack challenge `https://www.kaggle.com/datasets/google-brain/nips-2017-adversarial-learning-development-set`."""

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    root = DATASET_CACHE_DIR / "NIPS2017"

    def __init__(
        self,
        center: bool = False,
        size: int = 224,
    ) -> None:
        """Initialize the dataset.

        Args:
            - center: Normalize the images if needed.
            - size: Resize the images to size x size.
        """
        transf_list = []

        if center:
            transf_list.append(tsfm.Normalize(self.mean, self.std))

        transf_list.append(tsfm.Resize(size=(size, size), antialias=None))
        transf_list.append(tsfm.ToTensor())
        transform = tsfm.Compose(transf_list)

        super().__init__(root=self.root, transform=transform)
