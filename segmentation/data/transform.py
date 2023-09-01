import albumentations as albu
import albumentations.pytorch as albu_pytorch
from torchvision import transforms
from typing import Tuple

""" Data Augmentation Pipelines"""


def post_augmentation():
    post_transforms = [albu.Normalize(), albu_pytorch.transforms.ToTensorV2()]
    return albu.Compose(post_transforms)


def get_training_augmentation(image_size: Tuple[int, int]) -> albu.Compose:
    img_h, img_w = image_size
    train_transforms = [
        albu.ShiftScaleRotate(rotate_limit=25, border_mode=0, p=0.5),
        albu.PadIfNeeded(
            min_height=img_h, min_width=img_w, always_apply=True, border_mode=0
        ),
        albu.Resize(height=img_h, width=img_w, p=1),
        albu.GaussNoise(p=0.3),
        albu.OneOf(
            [
                albu.CLAHE(p=0.5),
                albu.RandomBrightnessContrast(p=0.6),
                albu.RandomGamma(p=0.4),
            ],
            p=0.8,
        ),
        albu.OneOf(
            [
                albu.Sharpen(p=0.5),
                albu.Blur(blur_limit=3, p=0.4),
                albu.MotionBlur(blur_limit=3, p=0.5),
            ],
            p=0.7,
        ),
        albu.HueSaturationValue(p=0.15),
        post_augmentation(),
    ]
    return albu.Compose(train_transforms)


def get_val_test_augmentation(image_size: Tuple[int, int]) -> albu.Compose:
    img_h, img_w = image_size
    val_test_transforms = [
        albu.PadIfNeeded(img_h, img_w),
        albu.Resize(height=img_h, width=img_w, p=1),
        post_augmentation(),
    ]
    return albu.Compose(val_test_transforms)


def get_base_transform_augmentation(image_size: Tuple[int, int]) -> albu.Compose:
    img_h, img_w = image_size
    transforms_list = [
        albu.Resize(height=img_h, width=img_w, p=1),
        post_augmentation(),
    ]
    return albu.Compose(transforms_list)
