import torch
from torch import nn
#from kornia.losses import binary_focal_loss_with_logits
from torchvision.ops.focal_loss import sigmoid_focal_loss
from ..loss import k_one_hot

class BinaryFocalLossWithLogits(nn.Module):
    """
    Difference from Kornia implementation is that
    1. It will convert labels to one hot if needed.
    2. Different default values (alpha = 0.25, reduction = mean)
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean') -> None:
        super().__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.dim() != input.dim():
            target = k_one_hot(target, input.size(-1))

        #return binary_focal_loss_with_logits(input, target, self.alpha, self.gamma, self.reduction)
        return sigmoid_focal_loss(input, target, self.alpha, self.gamma, self.reduction)
