import os
from PIL import Image
import torch

from pathlib import Path
from typing import Optional, Tuple

from torch.utils.data import DataLoader

from segmentation.data.transform import (
    get_training_augmentation,
    get_val_test_augmentation,
)
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid
from segmentation.data.datasets import lyft
import pytorch_lightning as pl


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: str,
        data_dir: str,
        batch_size: int,
        num_workers: Optional[int] = 1,
        image_size: Tuple[int, int] = (512, 512),
    ):
        super().__init__()
        self.dataset = dataset
        self.data_dir = data_dir
        self.dataset_colormap = lyft.DATASET_COLORS if self.dataset == "lyft" else []
        self.dataset_classnames = lyft.DATASET_LABELS if self.dataset == "lyft" else ()
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers else os.cpu_count()
        self.image_size = image_size

    def get_dataset(self, segment: str):
        augmentations = (
            get_training_augmentation(image_size=self.image_size)
            if segment == "train"
            else get_val_test_augmentation(image_size=self.image_size)
        )

        if self.dataset == "lyft":
            data_dirs_split = {"train": ["A", "B", "C"], "val": ["D"], "test": ["E"]}
            images_dirs_list = [
                Path(self.data_dir, f"data{part}", f"data{part}", "CameraRGB")
                for part in data_dirs_split[segment]
            ]
            masks_dirs_list = [
                Path(self.data_dir, f"data{part}", f"data{part}", "CameraSeg")
                for part in data_dirs_split[segment]
            ]
            # dataset = lyft.LyftDataset(
            #     images_dirs_list, masks_dirs_list, augmentations, debug_subset_size=10
            # ) # tmp for debug
            dataset = lyft.LyftDataset(images_dirs_list, masks_dirs_list, augmentations)
            print(f"[DATASET INFO] {self.dataset} - {segment} - {len(dataset)} samples")

        else:
            raise NotImplementedError

        return dataset

    def get_dataloader(self, segment: str, serial: bool = False) -> DataLoader:
        dataset = self.get_dataset(segment)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(segment == "train"),
            num_workers=0 if serial else self.num_workers,
            drop_last=(segment == "train"),
            persistent_workers=False if serial else self.num_workers > 0,
        )

    def train_dataloader(self):
        dataset = self.get_dataset("train")
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        dataset = self.get_dataset("val")
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        dataset = self.get_dataset("test")
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def predict_dataloader(self):
        dataset = self.get_dataset("test")
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def show_batch(self, segment: str = "train") -> Image.Image:
        colormap = torch.tensor(self.dataset_colormap, dtype=torch.uint8)
        dataloader = self.get_dataloader(segment)
        img, mask = next(iter(dataloader))
        mask = colormap[mask].moveaxis(-1, 1)
        img = ((img + 1) / 2 * 255).to(mask.dtype)
        return to_pil_image(make_grid(torch.stack([mask, img], 1).flatten(end_dim=1)))
