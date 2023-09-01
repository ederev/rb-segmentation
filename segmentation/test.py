from tqdm import tqdm
import torch
import segmentation_models_pytorch as smp
from random import seed
from typing import List, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
from segmentation.data.viz import viz_prediction


def testset_evaluation(
    inference_model,
    test_dataloader,
    criterion,
    num_classes: int,
    dataset_labels: Tuple[str],
    dataset_colors: List[List[int]],
    experiment_path: Path,
):
    with torch.no_grad():
        outputs = []
        test_loss = 0.0
        for batch in tqdm(test_dataloader):
            normalized_image, mask = batch
            output = inference_model(normalized_image)

            # visualize image / mask / prediction
            viz_prediction(
                normalized_image,
                mask,
                torch.argmax(output, 1).unsqueeze(1),
                dataset_colors,
                experiment_path,
            )

            # calculate metrics
            tp, fp, fn, tn = smp.metrics.get_stats(
                torch.argmax(output, 1).unsqueeze(1),
                mask.long().unsqueeze(1),
                mode="multiclass",
                num_classes=num_classes,
            )
            outputs.append({"tp": tp, "fp": fp, "fn": fn, "tn": tn})
            loss = criterion(output, mask.long())
            test_loss += loss.item()

        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])
        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise").item()
        print(f"Test Loss: {test_loss / len(test_dataloader)}")
        print(f"IoU: {iou}")

    metrics = np.round(
        torch.stack(
            [
                torch.mean(smp.metrics.recall(tp, fp, fn, tn, reduction=None), 0),
                torch.mean(
                    smp.metrics.false_positive_rate(tp, fp, fn, tn, reduction=None), 0
                ),
                torch.mean(
                    smp.metrics.false_negative_rate(tp, fp, fn, tn, reduction=None), 0
                ),
                torch.mean(smp.metrics.iou_score(tp, fp, fn, tn, reduction=None), 0),
            ]
        ).numpy(),
        3,
    )

    info = dict(
        {"Metrics": ["Recall", "FPR", "FNR", "IoU"]},
        **{
            label_name: metrics[:, label]
            for label, label_name in enumerate(dataset_labels)
        },
    )

    result_metrics = pd.DataFrame(info)
    result_metrics.to_csv(f"{experiment_path}/test_evaluation_metrics.csv")
