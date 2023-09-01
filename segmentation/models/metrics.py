from torchmetrics import Accuracy, MetricCollection, F1Score
from torchmetrics import JaccardIndex
from torch import Tensor


class PerClassF1(F1Score):
    def __init__(self, class_idx, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_idx = class_idx

    def update(self, preds: Tensor, target: Tensor, **kwargs):
        preds = preds[:, self.class_idx]
        target = (target == self.class_idx).long()
        super().update(preds, target)

    @classmethod
    def make_many(cls, class_names, *args, **kwargs):
        return {
            f"F1_{name}": cls(i, *args, **kwargs) for i, name in enumerate(class_names)
        }
