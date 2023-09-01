import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Tuple, Optional
import numpy as np
from pathlib import Path
import os
from PIL import Image
from torchvision import transforms


def rgb_to_hex(rgb):
    return "#%02x%02x%02x" % rgb


def display_image_and_annotation(
    img_path: str,
    mask_path: str,
    project_path: str,
    dataset_colors: List[List[int]],
    dataset_labels: Tuple[str, ...],
    save_fig: bool = False,
) -> None:
    """Drawing semantic segmentation mask and visualize it with original image and masked image.

    Args:
        img_path (str): _description_
        mask_path (str): _description_
        dataset_colors (List[List[int]]): _description_
        dataset_labels (Tuple[str, ...]): _description_
        save_fig (bool, optional): _description_. Defaults to False.
        show_plt (bool, optional): _description_. Defaults to False.
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(mask_path)
    rgb_mask = np.zeros_like(img)
    for label, color in enumerate(dataset_colors):
        rgb_mask[mask[:, :, 2] == label] = np.array(color)

    # blend image + rgb_mask
    masked_image = img * 0.5 + rgb_mask * 0.5
    masked_image = masked_image.astype(np.uint8)

    # plot
    fig, arr = plt.subplots(1, 4, figsize=(18, 5))

    arr[0].imshow(img)
    arr[0].set_title("RGB Image")
    arr[1].imshow(mask[:, :, 2])
    arr[1].set_title(f"Mask Labels - {len(np.unique(mask))}")
    arr[2].imshow(masked_image)
    arr[2].set_title("Image Masked")
    arr[3].imshow(rgb_mask)
    arr[3].set_title("RGB Mask")

    # create a legend with a color box
    legend_patchs = []
    for lbl in np.unique(mask):
        legend_patchs.append(
            mpatches.Patch(
                color=rgb_to_hex(tuple(dataset_colors[lbl])),
                label=f"{lbl} - {dataset_labels[lbl]}",
            )
        )

    arr[3].legend(
        handles=legend_patchs,
        bbox_to_anchor=(1.04, 0.5),
        loc="center left",
        framealpha=0.5,
        borderaxespad=0,
        frameon=True,
    )

    if save_fig:
        os.makedirs(Path(project_path, "dataset_samples"), exist_ok=True)
        fig.savefig(Path(project_path, "dataset_samples", f"{Path(img_path).stem}.jpg"))
        plt.close()
    else:
        plt.show()


def visualize_seg_mask(
    data_samples: List[np.ndarray],
    dataset_colors: List[List[int]],
    dataset_labels: Tuple[str, ...],
    main_title: Optional[str] = None,
) -> None:
    num_samples = len(data_samples)

    fig, axes_list = plt.subplots(nrows=num_samples, ncols=4, figsize=(15, 5))
    plt.subplots_adjust(hspace=0, wspace=0)

    orig_img, ann_mask, masked_img, rgb_mask = (
        axes_list[0][0],
        axes_list[0][1],
        axes_list[0][2],
        axes_list[0][3],
    )

    orig_img.set_title("Image", fontsize=10)
    ann_mask.set_title("Mask", fontsize=10)
    masked_img.set_title("Masked image", fontsize=10)
    rgb_mask.set_title("RGB Mask", fontsize=10)

    for idx in range(num_samples):
        image, mask = data_samples[idx]

        color_seg = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(dataset_colors):
            color_seg[mask == label, :] = color

        masked_image = np.array(image) * 0.5 + color_seg * 0.5
        masked_image = masked_image.astype(np.uint8)

        axes_list[idx][0].imshow(image)
        axes_list[idx][1].imshow(mask)
        axes_list[idx][2].imshow(masked_image)
        axes_list[idx][3].imshow(color_seg)

        axes_list[idx][0].set_axis_off()
        axes_list[idx][1].set_axis_off()
        axes_list[idx][2].set_axis_off()
        axes_list[idx][3].set_axis_off()

        # create a legend with a color box
        legend_patchs = []
        for lbl in np.unique(mask):
            legend_patchs.append(
                mpatches.Patch(
                    color=rgb_to_hex(tuple(dataset_colors[lbl])),
                    label=f"{lbl} - {dataset_labels[lbl]}",
                )
            )

        axes_list[idx][3].legend(
            ncol=len(legend_patchs),
            handles=legend_patchs,
            bbox_to_anchor=(1.04, 0.5),
            loc="center left",
            framealpha=0.5,
            borderaxespad=0,
            frameon=True,
        )

    if main_title:
        plt.suptitle(
            main_title,
            x=0.05,
            y=1.0,
            horizontalalignment="left",
            fontweight="semibold",
            fontsize="large",
        )
    plt.show()


def visualize_images_in_row(**images):
    """Plot images in one row."""
    n = len(images)
    plt.figure(figsize=(15, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())
        plt.imshow(image)
    plt.show()


def image_inverse_transform(norm_tensor):
    invTrans = transforms.Compose(
        [
            transforms.Normalize(
                mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
            ),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]),
        ]
    )

    inv_tensor = invTrans(norm_tensor)
    return inv_tensor


def viz_prediction(
    normalized_images, masks, pred_masks, dataset_colors, experiment_path
):
    palette = [value for color in dataset_colors for value in color]

    samples = normalized_images.shape[0]
    columns = ["Input Image", "Annotated mask", "Prediction"]
    fig, axes = plt.subplots(
        samples,
        3,
        figsize=(20, 10),
        sharex="row",
        sharey="row",
        subplot_kw={"xticks": [], "yticks": []},
        tight_layout=True,
    )

    for ax, column in zip(axes[0], columns):
        ax.set_title(column, fontsize=25)

    for i in range(samples):
        img = normalized_images[i]
        # denormalize image to rgb init
        img = image_inverse_transform(img).detach().numpy()
        m = masks[i].detach().numpy()
        pred_m = pred_masks[i].detach().squeeze(0)

        m = Image.fromarray(m).convert("P")
        pred_m = Image.fromarray(np.array(pred_m, np.int32)).convert("P")

        m.putpalette(palette)
        pred_m.putpalette(palette)

        axes[i][0].imshow(np.moveaxis(img, 0, -1))
        axes[i][1].imshow(m)
        axes[i][2].imshow(pred_m)

    fig.savefig(f"{experiment_path}/test_evaliation_samples.png")
    plt.close()
