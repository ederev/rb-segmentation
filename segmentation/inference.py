import cv2
from pathlib import Path
import torch
import albumentations as albu
import albumentations.pytorch as albu_pytorch
import numpy as np
from segmentation.data.transform import post_augmentation
from segmentation.data.datasets.lyft import DATASET_COLORS


def get_prediction(input_tensor, model):
    logits = model(input_tensor)
    preds = torch.argmax(logits, 1).squeeze(0)
    return preds


def process_frame(frame, model):
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    resized_image = albu.Resize(height=HEIGHT, width=WIDTH, p=1)(image=frame)["image"]
    image = post_augmentation()(image=resized_image)
    input_tensor = image["image"].unsqueeze(0)
    predicted_mask = get_prediction(input_tensor, model)
    # print(torch.unique(predicted_mask))

    mask = predicted_mask.numpy().astype(np.uint8)
    rgb_mask = np.zeros_like(resized_image)
    for label, color in enumerate(DATASET_COLORS):
        rgb_mask[mask == label] = np.array(color)

    # blend image + rgb_mask
    masked_image = resized_image * 0.5 + rgb_mask * 0.5
    masked_image = cv2.cvtColor(masked_image.astype(np.uint8), cv2.COLOR_RGB2BGR)

    res_frame = np.concatenate(
        [
            resized_image,
            masked_image,
        ],
        axis=1,
    )
    return res_frame


def process_video(video_source, model):
    vid_capture = cv2.VideoCapture(video_source)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    fps = vid_capture.get(5)
    print(f"frames per second : {fps} FPS")
    h, w = 512, 1024
    video_writer = cv2.VideoWriter("outputs/video.mp4", fourcc, fps, (w, h))

    # check if the webcam is opened correctly
    if not vid_capture.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        ret, frame = vid_capture.read()
        if not ret:
            break

        res_frame = process_frame(frame, model)
        cv2.imshow("result", res_frame)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break

        video_writer.write(res_frame)

    vid_capture.release()
    video_writer.release()
    cv2.destroyAllWindows()


HEIGHT, WIDTH = 512, 512

if __name__ == "__main__":
    video_source = "data/test_video.mp4"

    model_path = Path("outputs/UNet_timm-mobilenetv3_large_100/model_traced.pt")
    model = torch.jit.load(model_path)
    model.eval()

    process_video(video_source, model)
    print("finished")
