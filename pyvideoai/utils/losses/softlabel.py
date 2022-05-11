import torch
import torch.nn.functional as F
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


class MaskedSoftlabelRegressionLoss(nn.Module):
    """Similar to negative log likelihood,
    but we use sigmoid instead of softmax and we have soft labels
    When the target is negative value, that will be masked out before computing the loss.

    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps


    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        """
        Inputs:
            inputs: logits of shape (N, C).
            targets: targets of shape (N, C). Values lower than zero are considered as mask.

        Outputs:
            Loss: a scalar tensor, normalised by N.
        """
        assert targets.shape == inputs.shape, f'Unexpected shapes. {inputs.shape = } and {targets.shape = }'

        preds = torch.sigmoid(inputs)
        losses = []
        for pred, target_prob in zip(preds, targets):
            mask = target_prob >= 0. - self.eps
            pred = pred[mask]
            target_prob = target_prob[mask]

            loss = (target_prob * torch.log(pred+self.eps)) + ((1 - target_prob) * torch.log(1-pred+self.eps))
            loss = -loss.sum()
            losses.append(loss)

        return sum(losses) / len(losses)


class MaskedBinaryCrossEntropyLoss(nn.Module):
    """Same as MaskedSoftlabelRegressionLoss
    but numerically stable.
    """
    def __init__(self):
        super().__init__()


    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        """
        Inputs:
            inputs: logits of shape (N, C).
            targets: targets of shape (N, C). Values lower than zero are considered as mask.

        Outputs:
            Loss: a scalar tensor, normalised by N.
        """
        assert targets.shape == inputs.shape, f'Unexpected shapes. {inputs.shape = } and {targets.shape = }'

        #preds = torch.sigmoid(inputs)
        losses = []
        for input_logits, target_prob in zip(inputs, targets):
            mask = target_prob >= 0. - 1e-6
            input_logits = input_logits[mask]
            target_prob = target_prob[mask]

            loss = (target_prob * F.logsigmoid(input_logits)) + ((1 - target_prob) * F.logsigmoid(-input_logits))
            loss = -loss.sum()
            losses.append(loss)

        return sum(losses) / len(losses)
