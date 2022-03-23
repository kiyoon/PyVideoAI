import torch
from torch import Tensor

from .crossentropy import CrossEntropy
from ...exceptions import ParamException
import torch.nn.functional as F

import verboselogs 
logger = verboselogs.VerboseLogger(__name__)

EPS = 1e-6

class ProSelfLC(CrossEntropy):
    """
    IMPORTANT: I think this code is numerically instable?
    Kiyoon edit: forward() function inputs pred and labels only,
    and cur_time is going to be called as step().
    Call `step()` after forward.
    Example usage:

    ```
    loss = criterion(preds, labels) 
    criterion.step()
    ```

    Also, allow target to be of shape (N,) as well as (N,C). 
    Finally, take logits instead of probability distribution.
    ---

    The implementation for progressive self label correction (CVPR 2021 paper).
    The target probability will be corrected by
    a predicted distributions, i.e., self knowledge.
        1. ProSelfLC is partially inspired by prior related work,
            e.g., Pesudo-labelling.
        2. ProSelfLC is partially theorectically bounded by
            early stopping regularisation.

    Inputs: two tensors for predictions and target.
        1. predicted logits of shape (N, C)
        2. target probability  distributions of shape (N, C)
        3. current time (epoch/iteration counter).
        4. total time (total epochs/iterations)
        5. exp_base: the exponential base for adjusting epsilon
        6. counter: iteration or epoch counter versus total time.

    Outputs: scalar tensor, normalised by the number of examples.
    """

    def __init__(
        self, total_time: int, exp_base: float = 1, counter: str = "iteration"
    ) -> None:
        super().__init__()
        self.total_time = total_time
        self.exp_base = exp_base
        self.counter = counter
        self.epsilon = None
        self.cur_time = 0   # current time (epoch/iteration counter).

        if not (self.exp_base > 0):
            error_msg = (
                "self.exp_base = "
                + str(self.exp_base)
                + ". "
                + "The exp_base has to be larger than zero. "
            )
            raise (ParamException(error_msg))

        if not (isinstance(self.total_time, int) and self.total_time > 0):
            error_msg = (
                "self.counter = "
                + str(self.total_time)
                + ". "
                + "The counter has to be a positive integer. "
            )
            raise (ParamException(error_msg))

        if self.counter not in ["iteration", "epoch"]:
            error_msg = (
                "self.counter = "
                + str(self.counter)
                + ". "
                + "The counter has to be iteration or epoch. "
                + "The training time is counted by eithor of them. "
                + "The default option is iteration. "
            )
            raise (ParamException(error_msg))

    def update_epsilon_progressive_adaptive(self, pred_probs, cur_time):
        # global trust/knowledge
        time_ratio_minus_half = torch.tensor(cur_time / self.total_time - 0.5)
        global_trust = 1 / (1 + torch.exp(-self.exp_base * time_ratio_minus_half))
        # example-level trust/knowledge
        class_num = pred_probs.shape[1]
        H_pred_probs = torch.sum(-pred_probs * torch.log(pred_probs + 1e-6), 1)
        H_uniform = -torch.log(torch.tensor(1.0 / class_num))
        example_trust = 1 - H_pred_probs / H_uniform
        # the trade-off
        self.epsilon = global_trust * example_trust
        # from shape [N] to shape [N, 1]
        self.epsilon = self.epsilon[:, None]

    def forward(
        self, pred_logits: Tensor, target_probs: Tensor
    ) -> Tensor:
        """
        Inputs:
            1. predicted probability distributions of shape (N, C)
            2. target probability  distributions of shape (N, C), or just (N,).

        Outputs:
            Loss: a scalar tensor, normalised by N.
        """
        if not (self.cur_time <= self.total_time and self.cur_time >= 0):
            error_msg = (
                "The cur_time = "
                + str(self.cur_time)
                + ". The total_time = "
                + str(self.total_time)
                + ". The cur_time has to be no larger than total time "
                + "and no less than zero."
            )
            raise (ParamException(error_msg))

        if target_probs.dim() == 1:
            # Shape of (N,), not (N, C).
            # Convert it to one hot.
            num_classes = pred_logits.shape[1]
            target_probs = F.one_hot(target_probs, num_classes)

        # update self.epsilon
        pred_probs = F.softmax(pred_logits, -1)
        self.update_epsilon_progressive_adaptive(pred_probs, self.cur_time)

        logger.spam(f'Original label: {target_probs}')
        logger.spam(f'ProSelfLC pred weight: {self.epsilon}')
        new_target_probs = (1 - self.epsilon) * target_probs + self.epsilon * pred_probs
        logger.spam(f'ProSelfLC new label: {new_target_probs}')
        # reuse CrossEntropy's forward computation
        return super().forward(pred_probs, new_target_probs)

    def step(self, cur_time:int = None):
        if cur_time is None:
            self.cur_time += 1
        else:
            if not isinstance(cur_time, int):
                raise ValueError(f'cur_time has to be integer type but got {type(cur_time)}.')
            elif cur_time < 0:
                raise ValueError(f'cur_time has to be zero or bigger but got {cur_time}.')

            self.cur_time = cur_time


class MaskedProSelfLC(ProSelfLC):
    """
    It's the same as ProSelfLC in terms of label correction,
    but it will ignore some of the classes before doing label correction and softmax.
    It will ignore the ones with their label being negative (target_probs < 0)
    """

    def get_epsilon_progressive_adaptive_each_sample(self, pred_prob, pred_logprob, cur_time):
        # global trust/knowledge
        time_ratio_minus_half = torch.tensor(cur_time / self.total_time - 0.5)
        global_trust = 1 / (1 + torch.exp(-self.exp_base * time_ratio_minus_half))
        # example-level trust/knowledge
        class_num = pred_prob.shape[0]
        H_pred_probs = torch.sum(-pred_prob * pred_logprob, 0)
        H_uniform = -torch.log(torch.tensor(1.0 / class_num))
        example_trust = 1 - H_pred_probs / H_uniform
        # the trade-off
        epsilon = global_trust * example_trust
        return epsilon


    def forward(
        self, pred_logits: Tensor, target_probs: Tensor
    ) -> Tensor:
        """
        Inputs:
            1. predicted probability distributions of shape (N, C)
            2. target probability  distributions of shape (N, C), should be 0, 1 or -1. -1 for classes to be masked out.

        Outputs:
            Loss: a scalar tensor, normalised by N.
        """
        if not (self.cur_time <= self.total_time and self.cur_time >= 0):
            error_msg = (
                "The cur_time = "
                + str(self.cur_time)
                + ". The total_time = "
                + str(self.total_time)
                + ". The cur_time has to be no larger than total time "
                + "and no less than zero."
            )
            raise (ParamException(error_msg))

        if target_probs.dim() != 2:
            raise ValueError(f'target_probs has to be of shape (N, C) but got {target_probs.shape}')

        losses = []
        for pred_logit, target_prob in zip(pred_logits, target_probs):
            mask = target_prob >= 0. - EPS

            pred_logit = pred_logit[mask]
            target_prob = target_prob[mask]

            pred_prob = F.softmax(pred_logit, -1)
            pred_logprob = F.log_softmax(pred_logit, -1)
            epsilon = self.get_epsilon_progressive_adaptive_each_sample(pred_prob, pred_logprob, self.cur_time)
            new_target_prob = (1 - epsilon) * target_prob + epsilon * pred_prob
            loss = torch.sum(new_target_prob * (-pred_logprob), 0)
            losses.append(loss)

        num_examples = pred_logits.shape[0] # or len(losses)
        loss = sum(losses) / num_examples
        return loss
