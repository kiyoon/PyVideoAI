import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from ...exceptions import ParamException

class CrossEntropy(nn.Module):
    """
    The new implementation of cross entropy using two distributions.
    This can be a base class for other losses:
        1. label smoothing;
        2. bootsoft (self label correction), joint-soft,etc.
        3. proselflc
        ...

    Inputs: two tensors for predictions and target.
        1. predicted probability distributions of shape (N, C)
        2. target probability  distributions of shape (N, C)

    Outputs: scalar tensor, normalised by the number of examples.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred_probs: Tensor, target_probs: Tensor) -> Tensor:
        """
        Inputs:
            pred_probs: predictions of shape (N, C).
            target_probs: targets of shape (N, C).

        Outputs:
            Loss: a scalar tensor, normalised by N.
        """
        if not (pred_probs.shape == target_probs.shape):
            error_msg = (
                "pred_probs.shape = " + str(pred_probs.shape) + ". "
                "target_probs.shape = "
                + str(target_probs.shape)
                + ". "
                + "Their shape has to be identical. "
            )
            raise (ParamException(error_msg))
        # TODO: to assert values in the range of [0, 1]

        num_examples = pred_probs.shape[0]
        loss = torch.sum(target_probs * (-torch.log(pred_probs + 1e-6)), 1)
        loss = sum(loss) / num_examples
        return loss

class InstableCrossEntropy(CrossEntropy):
    def forward(self, pred_logits: Tensor, target: Tensor) -> Tensor:
        """
        Instead of probability distribution, it takes logits and label just like torch.nn.CrossEntropyLoss()
        Inputs:
            pred_logits: (N, C)
            target: (N,)
        """
        pred_probs = F.softmax(pred_logits, -1)
        num_classes = pred_logits.shape[1]
        target_probs = F.one_hot(target, num_classes)
        return super().forward(pred_probs, target_probs)


