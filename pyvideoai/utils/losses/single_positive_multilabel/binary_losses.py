import torch
from torch import nn
import torch.nn.functional as F
from ..loss import k_one_hot



class AssumeNegativeLossWithLogits(nn.Module):
    """
    Single positive multi-label setting:
    Labels are 1, -1 or 0, where 0 means not certain and -1 is negative.

    Assume Negative loss:
    For every 0, we treat it as negative labels.

    Numerical stability:
    log(1-sigmoid(logits)) equal to logsigmoid(-logits)
    """
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        if targets.dim() != inputs.dim():
            targets = k_one_hot(targets, inputs.size(-1))

        #preds = torch.sigmoid(inputs)
        #loss = (targets * torch.log(preds+self.eps)) + ((1 - targets) * torch.log(1-preds+self.eps))

        loss = (targets * F.logsigmoid(inputs)) + ((1 - targets) * F.logsigmoid(-inputs))

        return -loss.sum(dim=-1).mean(dim=0)


class WeakAssumeNegativeLossWithLogits(AssumeNegativeLossWithLogits):
    """
    A.K.A Down Weighting
    """
    def __init__(self, num_classes: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.gamma = 1 / (num_classes - 1)
        assert 0 <= self.gamma <= 1


    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        if targets.dim() != inputs.dim():
            targets = k_one_hot(targets, inputs.size(-1))

        #loss = (targets * torch.log(preds + self.eps)) + ((1 - targets) * self.gamma * torch.log(1 - preds + self.eps))
        loss = (targets * F.logsigmoid(inputs)) + ((1 - targets) * self.gamma * F.logsigmoid(-inputs))

        return -loss.sum(dim=-1).mean(dim=0)


class BinaryLabelSmoothLossWithLogits(AssumeNegativeLossWithLogits):
    def __init__(self, smoothing: float = 0.1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        assert 0 <= smoothing < 1
        self.smoothing = smoothing


    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        #if targets.dim() != 1:
        #    raise ValueError(f'Only support targets with positive integer labels, but got {targets.dim() = }')

        targets = k_one_hot(targets, inputs.size(-1), smoothing = self.smoothing, smooth_only_negatives=False)

        loss = (targets * F.logsigmoid(inputs)) + ((1 - targets) * F.logsigmoid(-inputs))

        return -loss.sum(dim=-1).mean(dim=0)


class BinaryNegativeLabelSmoothLossWithLogits(BinaryLabelSmoothLossWithLogits):
    """
    Label smoothing for only assumed negatives
    """
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        #if targets.dim() != 1:
        #    raise ValueError(f'Only support targets with positive integer labels, but got {targets.dim() = }')

        targets = k_one_hot(targets, inputs.size(-1), smoothing = self.smoothing, smooth_only_negatives=True)

        loss = (targets * F.logsigmoid(inputs)) + ((1 - targets) * F.logsigmoid(-inputs))

        return -loss.sum(dim=-1).mean(dim=0)
