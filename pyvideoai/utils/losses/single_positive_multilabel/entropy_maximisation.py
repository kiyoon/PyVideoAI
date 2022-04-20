import torch
import torch.nn.functional as F
from .binary_losses import AssumeNegativeLossWithLogits
from ..loss import k_one_hot



class EntropyMaximiseLossWithLogits(AssumeNegativeLossWithLogits):
    def __init__(self, alpha: float = 0.2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        assert 0 <= alpha <= 1
        self.alpha = alpha


    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        if targets.dim() != inputs.dim():
            targets = k_one_hot(targets, inputs.size(-1))

        preds = torch.sigmoid(inputs)
        logsig_pos = F.logsigmoid(inputs)
        logsig_neg = F.logsigmoid(-inputs)

        #entropy = -(preds * torch.log(preds + self.eps) + (1-preds) * torch.log(1-preds + self.eps))
        entropy = -(preds * logsig_pos + (1 - preds) * logsig_neg)
        loss = (targets * logsig_pos) + ((1 - targets) * self.alpha * entropy)
        return -loss.sum(dim=-1).mean(dim=0)
