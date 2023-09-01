import io

import torch
from PIL import Image
from torchvision import transforms
import albumentations as albu
import albumentations.pytorch as albu_pytorch


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


def get_segmentator(model_path="model_traced.pt"):
    model = torch.jit.load(model_path)
    model.eval()
    print("model ---- ok")
    return model


def post_augmentation():
    post_transforms = [albu.Normalize(), albu_pytorch.transforms.ToTensorV2()]
    return albu.Compose(post_transforms)


def get_segments(model, binary_image, max_size=512):
    input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
    resized_image = input_image.resize((max_size, max_size))

    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    # input_tensor = post_augmentation()(image=resized_image)["image"]
    input_tensor = preprocess(resized_image)
    input_batch = input_tensor.unsqueeze(
        0
    )  # create a mini-batch as expected by the model

    with torch.no_grad():
        output = model(input_batch)

    # torch.save(input_batch, "input_batch.pt")
    # torch.save(output, "output_tensor.pt")

    # print("output.shape", output.shape)
    output_predictions = torch.argmax(output, 1)
    output_predictions = output_predictions.squeeze(0)
    # print("unique classes in predictions:", torch.unique(output_predictions))

    # create a color palette, selecting a color for each class
    # palette = torch.tensor([2**25 - 1, 2**15 - 1, 2**21 - 1])
    # colors = torch.as_tensor([i for i in range(num_classes)])[:, None] * palette
    # colors = (colors % 255).numpy().astype("uint8")
    colors = torch.as_tensor(DATASET_COLORS).numpy().astype("uint8")

    # plot the semantic segmentation predictions of all classes in each color
    res_img = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(
        input_image.size
    )
    res_img.putpalette(colors)

    return res_img
