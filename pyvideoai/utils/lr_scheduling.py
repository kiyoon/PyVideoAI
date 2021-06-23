import torch
from typing import Iterable

def multistep_lr_nonlinear(optimiser, milestones: Iterable[int], factors: Iterable[float], last_epoch=-1):
    """Similar to torch.optim.lr_scheduler.MultiStepLR,
    but with this you can define the factors individually,
    so it doesn't need to be decreasing with a constant factor gamma.

    params:
        milestones (list): List of epoch indices. Must be increasing.
        factors (list): List of factors that will be multiplied to the initial lr. Note that it's NOT multiplicative to the last lr.

    returns:
        PyTorch LR scheduler.
    """

    def _nonlinear_step(current_epoch):
        #assert len(milestones) == len(factors)
        #assert len(milestones) > 0

        last_milestone = 0
        last_factor = 1.0
        for milestone, factor in zip(milestones, factors):
            assert milestone > last_milestone, "milestones have to be increasing"
            if current_epoch < milestone:
                return last_factor

            last_milestone, last_factor = milestone, factor

        return last_factor

    return torch.optim.lr_scheduler.LambdaLR(optimiser, lambda current_epoch: _nonlinear_step(current_epoch), last_epoch=last_epoch)


def multistep_lr_nonlinear_iters(optimiser, iters_per_epoch, milestones: Iterable[int], factors: Iterable[float], last_epoch=-1):
    """Same as multistep_lr_nonlinear, but milestones will be multiplied by the iters_per_epoch.
    """

    milestones = list(map(lambda x: x*iters_per_epoch, milestones))

    return multistep_lr_nonlinear(optimiser, milestones, factors, last_epoch)
