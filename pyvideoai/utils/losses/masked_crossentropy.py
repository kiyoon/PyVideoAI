import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from .proselflc.exceptions import ParamException
EPS = 1e-6

class MaskedCrossEntropy(nn.Module):
    """
    Edit by Kiyoon from ProSelfLC:
        Implemented class masking.
        target_probs < 0 will be masked out and will not contribute to the loss.
        Also, they don't take part in when calculating softmax.

    Inputs: two tensors for predictions and target.
        1. predicted probability distributions of shape (N, C)
        2. target probability  distributions of shape (N, C)

    Outputs: scalar tensor, normalised by the number of examples.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred_logits: Tensor, target_probs: Tensor) -> Tensor:
        """
        Inputs:
            pred_logits: logits of shape (N, C).
            target_probs: targets of shape (N, C). Values lower than zero are considered as mask.

        Outputs:
            Loss: a scalar tensor, normalised by N.
        """
        if not (pred_logits.shape == target_probs.shape):
            error_msg = (
                "pred_logits.shape = " + str(pred_logits.shape) + ". "
                "target_probs.shape = "
                + str(target_probs.shape)
                + ". "
                + "Their shape has to be identical. "
            )
            raise (ParamException(error_msg))

        losses = []
        for pred_logit, target_prob in zip(pred_logits, target_probs):
            mask = target_prob >= 0. - EPS

            pred_logit = pred_logit[mask]
            target_prob = target_prob[mask]

            pred_prob = F.log_softmax(pred_logit, -1)
            loss = torch.sum(target_prob * (-pred_prob), 0)
            losses.append(loss)

        num_examples = pred_logits.shape[0] # or len(losses)
        loss = sum(losses) / num_examples
        return loss
