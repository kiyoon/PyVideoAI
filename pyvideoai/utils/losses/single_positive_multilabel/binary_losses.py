import torch
from kornia.losses import BinaryFocalLossWithLogits
from ..softlabel import SoftlabelRegressionLoss
from ..loss import k_one_hot



class AssumeNegativeLoss(SoftlabelRegressionLoss):
    pass


class WeakAssumeNegativeLoss(AssumeNegativeLoss):
    """
    A.K.A Down Weighting
    """
    def __init__(self, *args, num_classes: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.gamma = 1 / (num_classes - 1)
        assert 0 <= self.gamma <= 1


    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        if targets.dim() != inputs.dim():
            targets = k_one_hot(targets, inputs.size(-1))

        preds = torch.sigmoid(inputs)

        loss = (targets * torch.log(preds + self.eps)) + ((1 - targets) * self.gamma * torch.log(1 - preds + self.eps))

        return -loss.sum(dim=-1).mean(dim=0)


class BinaryLabelSmoothLoss(AssumeNegativeLoss):
    def __init__(self, *args, smoothing: float = 0.1, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        assert 0 <= smoothing < 1
        self.smoothing = smoothing


    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        if targets.dim() != 1:
            raise ValueError(f'Only support targets with positive integer labels, but got {targets.dim() = }')

        targets = k_one_hot(targets, inputs.size(-1), smoothing = self.smoothing, smooth_only_negatives=False)

        preds = torch.sigmoid(inputs)

        loss = (targets * torch.log(preds + self.eps)) + ((1 - targets) * self.gamma * torch.log(1 - preds + self.eps))

        return -loss.sum(dim=-1).mean(dim=0)


class BinaryNegativeLabelSmoothLoss(BinaryLabelSmoothLoss):
    """
    Label smoothing for only assumed negatives
    """
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        if targets.dim() != 1:
            raise ValueError(f'Only support targets with positive integer labels, but got {targets.dim() = }')

        targets = k_one_hot(targets, inputs.size(-1), smoothing = self.smoothing, smooth_only_negatives=True)

        preds = torch.sigmoid(inputs)

        loss = (targets * torch.log(preds + self.eps)) + ((1 - targets) * self.gamma * torch.log(1 - preds + self.eps))

        return -loss.sum(dim=-1).mean(dim=0)
