r"""Raw class for loading the ImageNet dataset with target labels."""

import csv
import shutil
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Optional

import kagglehub
import torch
from PIL import Image
from PIL.PngImagePlugin import PngImageFile


class ImageNetT(torch.utils.data.Dataset):
    r"""Imagenet datast from NeuriPS 2017 competition with target labels."""

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        r"""Initialize the ImageNet dataset.

        Args:
            root (str): Root folder of the dataset.
            transform (Callable): Transform to apply to the images.
            target_transform (Callable): Transform to apply to the target labels.

        """
        self.root = root
        self.img_paths, self.labels, self.target_labels = self._load_data()
        self.transform = transform
        self.target_transform = target_transform

    def download_dataset(self) -> None:
        r"""Download the dataset from kaggle."""
        dataset_link = "google-brain/nips-2017-adversarial-learning-development-set"
        path = kagglehub.dataset_download(dataset_link)
        # copy the images to the root folder
        shutil.copytree(path, self.root)
        # remove the downloaded folder
        shutil.rmtree(path)

    def _load_data(self) -> tuple[list[str], list[int], list[int]]:
        r"""Load the ImageNet dataset with target labels.

        Resized from 299x299 to 224x224.

        Args:
            dataset_root (str): Root folder of dataset.

        Return:
            list[str]: The paths of images.
            list[int]: The ground truth label of images.
            list[int]: The target label of images.

        """
        img_dir_paths = Path(self.root + "/images")
        # if  empty, download the dataset
        if not img_dir_paths.exists():
            self.download_dataset()
        img_paths = sorted(img_dir_paths.glob("*.png"))
        csv_path = Path(self.root + "/images.csv")
        gt_dict = defaultdict(int)
        tgt_dict = defaultdict(int)
        # Read the csv file and store the labels in a dictionary
        with csv_path.open() as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                gt_dict[row["ImageId"]] = int(row["TrueLabel"])
                tgt_dict[row["ImageId"]] = int(row["TargetClass"])
        gt_labels = [gt_dict[key] - 1 for key in sorted(gt_dict)]  # zero indexed
        tgt_labels = [tgt_dict[key] - 1 for key in sorted(tgt_dict)]  # zero indexed
        return img_paths, gt_labels, tgt_labels

    def __len__(self) -> int:
        r"""Return the lenght of the dataset."""
        return len(self.labels)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor | PngImageFile, torch.Tensor | int, torch.Tensor | int]:
        r"""Get the image, label and target label of the dataset."""
        img_path = self.img_paths[idx]
        image = Image.open(str(img_path))
        label = self.labels[idx]
        target = self.target_labels[idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label, target

    def __repr__(self) -> str:
        r"""Return the string representation of the dataset."""
        return (
            f"ImageNetT(root={self.root}, transform={self.transform}, "
            f"target_transform={self.target_transform})"
        )
