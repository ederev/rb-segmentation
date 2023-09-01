import os
from pathlib import Path
from typing import List, Union


def list_image_paths(dir_paths: List[Path]) -> List[Path]:
    image_paths = []
    for dir_path in dir_paths:
        # counter = 0
        image_filenames = os.listdir(dir_path)
        for image_filename in image_filenames:
            image_paths.append(dir_path / image_filename)
            # counter += 1
        # print(f"[FULL DATASET INFO] {dir_path} - {counter} samples")
    return image_paths
