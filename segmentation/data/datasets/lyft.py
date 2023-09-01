import random
import os
import cv2

from pathlib import Path
from typing import List, Optional

from torch.utils.data import Dataset
import albumentations as albu

from segmentation.data.utils import list_image_paths
from segmentation.data.viz import (
    display_image_and_annotation,
    visualize_seg_mask,
)

DATASET_LABELS = (
    "Unlabeled",
    "Building",
    "Fence",
    "Other",
    "Pedestrian",
    "Pole",
    "Roadline",
    "Road",
    "Sidewalk",
    "Vegetation",
    "Car",
    "Wall",
    "Traffic sign",
)

DATASET_COLORS = [
    [0, 0, 0],
    [70, 70, 70],
    [190, 153, 153],
    [250, 170, 160],
    [220, 20, 60],
    [153, 153, 153],
    [157, 234, 50],
    [128, 64, 128],
    [224, 35, 232],
    [107, 142, 35],
    [0, 0, 142],
    [102, 102, 156],
    [220, 220, 0],
]


class LyftDataset(Dataset):
    """LyftDataset Dataset class

    Args:
        Dataset (_type_): _description_
    """

    def __init__(
        self,
        images_dir: List[Path],
        masks_dir: List[Path],
        transform: Optional[albu.Compose] = None,
        debug_subset_size: Optional[int] = None,
    ):
        """_summary_

        Args:
            images_dir (List[Path]): paths to dirs with images
            masks_dir (List[Path]): paths to dirs with annotated segmentation masks
            transform (Optional[albu.Compose], optional): composed list of data augmentation functions from albumentations. Defaults to None.
            debug_subset_size (Optional[int], optional): number of samples for Debug. Defaults to None.
        """
        images_paths_list = list_image_paths(images_dir)
        masks_paths_list = list_image_paths(masks_dir)
        self.images = sorted(images_paths_list)
        self.masks = sorted(masks_paths_list)
        self.transform = transform

        if debug_subset_size:
            self.images = self.images[:debug_subset_size]
            self.masks = self.masks[:debug_subset_size]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_id = self.images[idx]
        image = cv2.imread(str(image_id))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_id = self.masks[idx]
        mask = cv2.imread(str(mask_id))[:, :, 2]

        if self.transform:
            sample = self.transform(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return image, mask


def display_random_data_samples(
    image_dirs_paths: List[Path],
    mask_dirs_paths: List[Path],
    proj_path: str,
    savefig: bool = False,
) -> None:
    """
    Display image and annotation mask
        for random sample from each folder A/B/C/D/E
    """
    num_folders = len(image_dirs_paths)
    for i in range(num_folders):
        img_path = str(image_dirs_paths[i])
        mask_path = str(mask_dirs_paths[i])

        # get random img name from current dir
        img_name = random.choice(os.listdir(img_path))
        img_path = os.path.join(img_path, img_name)
        mask_path = os.path.join(mask_path, img_name)

        # imshow image and mask
        display_image_and_annotation(
            img_path=img_path,
            mask_path=mask_path,
            project_path=proj_path,
            dataset_colors=DATASET_COLORS,
            dataset_labels=DATASET_LABELS,
            save_fig=savefig,
        )


if __name__ == "__main__":
    # (.venv) ekaterinaderevyanka@Ekaterinas-MacBook-Pro rb-segmentation % python -m segmentation.data.datasets.lyft

    DATA_DIR = "data/"
    PROJECT_PATH = "outputs"

    IMAGE_DIRS_PATHS = [
        Path(DATA_DIR, f"data{part}", f"data{part}", "CameraRGB")
        for part in ["A", "B", "C", "D", "E"]
    ]
    MASK_DIRS_PATHS = [
        Path(DATA_DIR, f"data{part}", f"data{part}", "CameraSeg")
        for part in ["A", "B", "C", "D", "E"]
    ]

    display_random_data_samples(
        IMAGE_DIRS_PATHS, MASK_DIRS_PATHS, PROJECT_PATH, savefig=True
    )

    data_dirs_split = {"train": ["A", "B", "C"], "val": ["D"], "test": ["E"]}
    # train
    IMAGES_TRAIN_LIST = [
        Path(DATA_DIR, f"data{part}", f"data{part}", "CameraRGB")
        for part in data_dirs_split["train"]
    ]
    MASKS_TRAIN_LIST = [
        Path(DATA_DIR, f"data{part}", f"data{part}", "CameraSeg")
        for part in data_dirs_split["train"]
    ]
    # val
    IMAGES_VAL_LIST = [
        Path(DATA_DIR, f"data{part}", f"data{part}", "CameraRGB")
        for part in data_dirs_split["val"]
    ]
    MASKS_VAL_LIST = [
        Path(DATA_DIR, f"data{part}", f"data{part}", "CameraSeg")
        for part in data_dirs_split["val"]
    ]
    # test
    IMAGES_TEST_LIST = [
        Path(DATA_DIR, f"data{part}", f"data{part}", "CameraRGB")
        for part in data_dirs_split["test"]
    ]
    MASKS_TEST_LIST = [
        Path(DATA_DIR, f"data{part}", f"data{part}", "CameraSeg")
        for part in data_dirs_split["test"]
    ]

    # testing datasets
    SUBSET_SIZE = 100
    train_dataset = LyftDataset(
        IMAGES_TRAIN_LIST, MASKS_TRAIN_LIST, debug_subset_size=SUBSET_SIZE
    )
    val_dataset = LyftDataset(
        IMAGES_VAL_LIST, MASKS_VAL_LIST, debug_subset_size=SUBSET_SIZE
    )
    test_dataset = LyftDataset(
        IMAGES_TEST_LIST, MASKS_TEST_LIST, debug_subset_size=SUBSET_SIZE
    )

    visualize_seg_mask(
        data_samples=[
            train_dataset[random.randint(0, SUBSET_SIZE) + i] for i in range(2)
        ],
        dataset_colors=DATASET_COLORS,
        dataset_labels=DATASET_LABELS,
        main_title="Training samples",
    )

    visualize_seg_mask(
        data_samples=[
            val_dataset[random.randint(0, SUBSET_SIZE) + i] for i in range(2)
        ],
        dataset_colors=DATASET_COLORS,
        dataset_labels=DATASET_LABELS,
        main_title="Validation samples",
    )

    visualize_seg_mask(
        data_samples=[
            test_dataset[random.randint(0, SUBSET_SIZE) + i] for i in range(2)
        ],
        dataset_colors=DATASET_COLORS,
        dataset_labels=DATASET_LABELS,
        main_title="Test samples",
    )
