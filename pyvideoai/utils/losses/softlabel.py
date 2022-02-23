import torch
from torch import nn
from .loss import k_one_hot
class SoftlabelRegressionLoss(nn.Module):
    """Similar to negative log likelihood,
    but we use sigmoid instead of softmax and we have soft labels
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps


    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        if targets.dim() != inputs.dim():
            targets = k_one_hot(targets, inputs.size(-1))

        preds = torch.sigmoid(inputs)

        loss = (targets * torch.log(preds+self.eps)) + ((1 - targets) * torch.log(1-preds+self.eps))

        return -loss.sum(dim=-1).mean(dim=0)
