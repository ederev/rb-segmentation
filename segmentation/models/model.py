from torchmetrics import Accuracy, MetricCollection
from torchmetrics import JaccardIndex
from torch import Tensor
import torch
from typing import Tuple, Optional, Callable
import segmentation_models_pytorch as smp

import pytorch_lightning as pl


class Model(pl.LightningModule):
    def __init__(
        self,
        model: Callable,
        optimizer,
        criterion,
        class_names: Tuple[str, ...],
        image_size: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        if self.hparams.image_size is not None:
            self.example_input_array = torch.rand(
                1,
                3,
                self.hparams.image_size[0],
                self.hparams.image_size[1],
                device=self.device,
            )

        self.class_names = class_names
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        self.val_metrics = MetricCollection(
            {
                "JaccardIndex": JaccardIndex(
                    task="multiclass", num_classes=len(self.class_names)
                ),
                "Accuracy": Accuracy(
                    task="multiclass", num_classes=len(self.class_names)
                ),
            },
            prefix="val/",
        )

    def forward(self, image: Tensor):
        return self.model(image)

    def shared_step(self, batch: Tuple[Tensor, Tensor], stage: str):
        image, mask = batch

        # print("image.shape", image.shape, mask.shape)
        # (batch_size, num_channels, height, width)
        assert image.ndim == 4

        logits = self.forward(image)
        loss = self.criterion(logits, mask.long())

        # true positive, false positive,
        # false negative, true negative
        # 'pixels' for each image and class
        tp, fp, fn, tn = smp.metrics.get_stats(
            torch.argmax(logits, 1).unsqueeze(1),
            mask.long().unsqueeze(1),
            mode="multiclass",
            num_classes=len(self.class_names),
        )

        # per image IoU - calculate IoU score for each image
        # and then compute mean over these scores
        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro-imagewise")

        # dataset IoU - aggregate intersection and union over whole dataset
        # and then compute IoU score
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        self.log(f"{stage}_IoU", iou, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_dataset_iou", dataset_iou, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_loss", loss)

        if stage == "valid":
            metrics = self.val_metrics
            preds = logits.softmax(1)
            metrics(preds, mask)
            self.log_dict(metrics)

        return {"loss": loss, "iou": iou}

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def configure_optimizers(self):
        return self.optimizer
