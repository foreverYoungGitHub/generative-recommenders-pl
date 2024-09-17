import torch
import torchmetrics
import torchmetrics.utilities


class MultiClsMetrics(torchmetrics.Metric):
    """
    A metric class for computing various classification metrics for rating predictions
    in ranking tasks.

    This class calculates AUROC (Area Under the Receiver Operating Characteristic),
    Average Precision, Precision, and Recall for multi-class rating predictions.

    Args:
        num_classes (int): The number of rating classes.
        **kwargs: Additional keyword arguments to pass to the parent Metric class.

    Attributes:
        preds (list): State to store prediction tensors.
        target (list): State to store target tensors.

    Methods:
        update(preds, target): Update the metric states with new predictions and targets.
        compute(): Compute and return all the classification metrics.
    """

    def __init__(self, num_classes: int, **kwargs):
        super().__init__()  # tmp remove kwargs to avoid error
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

        self.auroc = torchmetrics.AUROC(num_classes=num_classes, task="multiclass")
        self.ap = torchmetrics.AveragePrecision(
            num_classes=num_classes, task="multiclass"
        )
        self.precision = torchmetrics.Precision(
            num_classes=num_classes, task="multiclass"
        )
        self.recall = torchmetrics.Recall(num_classes=num_classes, task="multiclass")

    def update(self, preds: torch.Tensor, target: torch.Tensor, **kwargs):
        self.preds.append(preds)
        self.target.append(target)

    def compute(self):
        # Concatenate the lists of tensors
        preds = torchmetrics.utilities.dim_zero_cat(self.preds)
        target = torchmetrics.utilities.dim_zero_cat(self.target)

        output = {}
        output["auroc"] = self.auroc(preds, target)
        output["ap"] = self.ap(preds, target)
        output["precision"] = self.precision(preds, target)
        output["recall"] = self.recall(preds, target)
        return output

    def reset(self):
        super().reset()
        self.auroc.reset()
        self.ap.reset()
        self.precision.reset()
        self.recall.reset()
