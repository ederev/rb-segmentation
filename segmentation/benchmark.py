import time
import numpy as np

import torch.backends.cudnn as cudnn
import torch
import segmentation_models_pytorch as smp

from collections import OrderedDict
from typing import Tuple

cudnn.benchmark = True
import tqdm


def benchmark(
    model,
    device: str = "cuda",
    input_shape: Tuple[int, int, int, int] = (1, 3, 512, 512),
    nwarmup: int = 50,
    nruns: int = 100,
):
    input_data = torch.randn(input_shape)
    input_data = input_data.to(device)

    print("Warm up ...")
    with torch.no_grad():
        for _ in range(nwarmup):
            features = model(input_data)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print("Start timing ...")
    timings = []
    with torch.no_grad():
        for i in range(1, nruns + 1):
            start_time = time.time()
            features = model(input_data)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
            if i % 10 == 0:
                print(
                    "Iteration %d/%d, ave batch time %.2f ms"
                    % (i, nruns, np.mean(timings) * 1000)
                )

    print("Input shape:", input_data.size())
    print("Output features size:", features.size())
    print("Average batch time: %.2f ms" % (np.mean(timings) * 1000))


if __name__ == "__main__":
    checkpoint_model_path = "outputs/UNet_timm-mobilenetv3_large_100/checkpoints_UNet_timm-mobilenetv3_large_100/UNet_timm-mobilenetv3_large_100-v9.ckpt"
    traced_model_path = "outputs/UNet_timm-mobilenetv3_large_100/model_traced.pt"

    torch_model = smp.create_model(
        "UNet",
        encoder_name="timm-mobilenetv3_large_100",
        encoder_weights="imagenet",
        in_channels=3,
        classes=13,
    )

    device = None if torch.cuda.is_available() else torch.device("cpu")
    state_dict = torch.load(checkpoint_model_path, map_location=device)["state_dict"]
    lightning_state_dict = OrderedDict(
        [(key[6:], state_dict[key]) for key in state_dict.keys()]
    )
    torch_model.load_state_dict(lightning_state_dict)
    torch_model.eval()

    # CPU benchmarks
    # torch model
    print("=== CPU benchmarks ===")
    print("--- torch model ---")
    benchmark(torch_model, device="cpu")

    # traced model
    print("--- traced model ---")
    traced_model = torch.jit.load(traced_model_path).to("cpu")
    traced_model.eval()
    benchmark(traced_model, device="cpu")

    # CUDA benchmarks
    if torch.cuda.is_available():
        print("=== GPU benchmarks ===")
        # torch model
        print("--- torch model ---")
        torch_model = torch_model.to("cuda")
        benchmark(torch_model)

        # traced model
        print("--- traced model ---")
        traced_model = torch.jit.load(traced_model).to("cuda")
        traced_model.eval()
        benchmark(traced_model)
