import torch
from torch import nn
import torch.nn.functional as F
class SoftlabelRegressionLoss(nn.Module):
    """Similar to negative log likelihood,
    but we use sigmoid instead of softmax and we have soft labels
    """
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        preds = F.sigmoid(inputs)

        loss = (targets * torch.log(preds)) + ((1 - targets) * torch.log(1-preds))

        return -loss.sum(dim=-1)
