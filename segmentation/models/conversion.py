import onnx
import onnxruntime
import torch
import cv2
from pathlib import Path
import numpy as np
from segmentation.data.transform import get_base_transform_augmentation
import matplotlib.pyplot as plt


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


def export_trained_model(trained_model, experiment_path, input_size):
    trained_model.eval()
    for m in trained_model.modules():
        m.requires_grad = False

    input = torch.rand([1, 3, input_size[0], input_size[1]])
    torch_output = trained_model(input)
    print("torch_output.shape", torch_output.shape)

    # state dict
    pth_model_path = Path(experiment_path, "model_statedict.pth")
    torch.save(trained_model.state_dict(), pth_model_path)
    print(f"state-dict - saved {pth_model_path}")

    # convert to TorchScript
    try:
        traced_module = torch.jit.trace(trained_model, input)
        pt_model_path = Path(experiment_path, "model_traced.pt")
        torch.jit.save(traced_module, pt_model_path)
        print(f"traced jit - saved {pt_model_path}")

        # check jit
        pt_model = torch.jit.load(pt_model_path)
        pt_model.eval()
        torchscript_output = pt_model(input)
        print("torchscript_output.shape", torchscript_output.shape)

        # assert that the results are similar
        torch.testing.assert_allclose(torch_output, torchscript_output)
        print("assert_allclose torch_output vs torchscript_output - check")

    except Exception as exc:
        print(exc, " --- [ERROR] traced jit")

    try:
        # convert to onnx
        onnx_model_path = Path(experiment_path, "model_export_onnx.onnx")
        torch.onnx.export(
            trained_model,
            input,
            onnx_model_path,
            input_names=["input"],
            output_names=["output"],
            verbose=True,
            opset_version=17,
        )
        print("export onnx - saved onnx_model_path")

        # check onnx
        onnx_model = onnx.load(onnx_model_path)
        onnx.checker.check_model(onnx_model)
        # load the model using onnxruntime
        ort_session = onnxruntime.InferenceSession(onnx_model_path)
        input_name = ort_session.get_inputs()[0].name
        ort_inputs = {input_name: to_numpy(input)}
        ort_outs = ort_session.run(None, ort_inputs)
        print("ort_outs.shape", to_numpy(torch_output).shape)
        # compare ONNX Runtime and PyTorch results
        np.testing.assert_allclose(
            to_numpy(torch_output), ort_outs[0], rtol=1e-03, atol=1e-03
        )
        print(
            "Exported model has been tested with ONNX Runtime, and the result looks good!"
        )
        print("all models saved")
    except Exception as exc:
        print(exc, " --- [ERROR] onnx")


def check_traced_model(img_path, model_path, experiment_path, input_size=(512, 512)):
    model = torch.jit.load(model_path)
    model.eval()

    image = cv2.imread(str(img_path))
    # preprocess
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # resize
    # normalize and to tensor
    image = get_base_transform_augmentation(input_size)(image=image)
    image = image["image"].unsqueeze(0)
    print(image.shape)

    logits = model(image)
    preds = torch.argmax(logits, 1)
    print(np.unique(preds.detach().numpy()))

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    axes[0].imshow(image.squeeze(0).movedim(0, -1).detach().numpy())
    axes[1].imshow(preds.squeeze(0).detach().numpy())
    fig.tight_layout()
    plt.axis("off")
    fig.savefig(f"{experiment_path}/check_traced_model.png")
    plt.close()
