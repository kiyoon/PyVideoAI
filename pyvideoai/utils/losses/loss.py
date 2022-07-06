import sklearn
import numpy as np
import os

def compute_balanced_class_weight_from_csv(csv_path, label_column, num_classes, csv_separator=" ", label_separator=",", ignore_first_row=False):
    """
    params:
        csv_path (str): path to CSV
        label_column (int): column index (starting with 0) for the labels
        label_separator (str): if multilabel, separator for the labels
    """
    assert os.path.exists(csv_path), "{} not found".format(
        csv_path
    )
    y = []
    with open(csv_path, "r") as f:
        if ignore_first_row:
            f.readline()
        for line in f.read().splitlines():
            line_list = line.split(csv_separator)
            label = line_list[label_column]
            label_list = list(map(int, label.split(label_separator)))
            y.extend(label_list)
    return compute_balanced_class_weight(y, num_classes)


def compute_balanced_class_weight(y, num_classes):
    """Generate weights for loss for imbalanced dataset.
    Same as sklearn.utils.class_weight.compute_class_weight,
    but it will return "0" for the empty classes so that the length of the wieghts is `num_classes`
    """
    classes = np.unique(y)
    assert len(classes) <= num_classes, f"There are more classes than num_classes: {num_classes}"
    assert classes[-1] < num_classes, f"There are more classes ({classes[-1]+1}) than num_classes: ({num_classes})"
    weight = sklearn.utils.class_weight.compute_class_weight('balanced',
            classes = classes,
            y = y)
    if len(classes) == num_classes and classes[0] == 0 and classes[-1] == num_classes-1:
        # classes = [0, 1, 2, .., n-1]
        return weight
    else:
        extended_weight = np.zeros(num_classes, dtype=np.float)
        for class_idx, weight_val in zip(classes, weight):
            extended_weight[class_idx] = weight_val
        return extended_weight  # zero if no sample


"""
Kiyoon implementation of Label Smoothing Cross Entropy Loss taken from `https://stackoverflow.com/questions/55681502/label-smoothing-in-pytorch`
Edits:
    1. Apply official label smoothing formula from the paper. With smoothing=a and num_classes=K, y^LS = y(1-a) + a/K. True label becomes something like 0.933 when a=0.1, depending on how many classes you have.
        a. The original code from the URL implements differently. y^LS = y(1-a) + (1-y)*a/(K-1). True label becomes 0.9 when a=0.1
    2. Accepts custom smooth label instead of 1D tensor label. (OneHotCrossEntropyLoss).
        a. Make sure to provide labels that sum to 1.
"""
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from torch.nn.modules.loss import _WeightedLoss

class OneHotCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean'):
        super().__init__(weight=weight, reduction=reduction)
        self.weight = weight
        self.reduction = reduction

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
            if self.reduction == 'sum' else loss

    def forward(self, inputs, targets):
        log_preds = F.log_softmax(inputs, -1)

        if self.weight is not None:
            log_preds = log_preds * self.weight.unsqueeze(0)

        return self.reduce_loss(-(targets * log_preds).sum(dim=-1))

def k_one_hot(targets:torch.Tensor, n_classes:int, smoothing=0.0, smooth_only_negatives = False):
    with torch.no_grad():
        if targets.dim() == 1:
            targets = torch.empty(size=(targets.size(0), n_classes),
                                  device=targets.device) \
                                  .fill_(smoothing / n_classes) \
                                  .scatter_(1, targets.data.unsqueeze(1), 1. if smooth_only_negatives else 1. - smoothing + smoothing / n_classes)
        elif targets.dim() == 2:
            # WARNING: it only works correctly with single labels even though it's 2D tensor.
            # If the labels are multi-labels, it would be incorrect.
            assert n_classes == targets.size(1), f'number of classes does not match with the tensor and the argument. {n_classes = } and {targets.size(1) = }'
            if smooth_only_negatives:
                # 0 -> smoothing/n_classes
                # 1 -> 1
                # f(x) = x + (1-x)*(smoothing/n_classes)
                targets = targets + (1.-targets) * (smoothing/n_classes)
            else:
                # 0 -> smoothing/n_classes
                # 1 -> 1 - smoothing + smoothing/n_classes
                # f(x) = smoothing/n_classes + x*(1-smoothing)
                targets = targets * (1. - smoothing) + smoothing/n_classes
        else:
            raise ValueError(f'targets.dim() should be 1 or 2 but got {targets.dim()}')
    return targets


class LabelSmoothCrossEntropyLoss(OneHotCrossEntropyLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing


    def forward(self, inputs, targets):
        assert 0 <= self.smoothing < 1

        targets = k_one_hot(targets, inputs.size(-1), self.smoothing)
        return super().forward(inputs, targets)



if __name__ == '__main__':
    crit = CrossEntropyLoss()
    predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
				 [0, 0.9, 0.2, 0.2, 1],
				 [1, 0.2, 0.7, 0.9, 1]])
    label = torch.LongTensor([2, 1, 0])
    onehot_label = torch.FloatTensor([[0., 0., 1., 0., 0.],
				 [0., 1., 0., 0., 0.],
				 [1., 0, 0, 0, 0]])

    # Official PyTorch CrossEntropyLoss test with 1D tensor labels.
    v = crit(Variable(predict),
	     Variable(label))
    print(f'Official PyTorch CrossEntropyLoss: {v}')

    # OneHotCrossEntropyLoss test with one-hot labels
    crit = OneHotCrossEntropyLoss()
    v = crit(Variable(predict),
	     Variable(onehot_label))
    print(f'OneHotCrossEntropyLoss: {v}')

    # OneHotCrossEntropyLoss test with custom applied smooth labels
    smooth_label = k_one_hot(label, 5, smoothing=0.3)
    print(f'{smooth_label = }')
    v = crit(Variable(predict),
	     Variable(smooth_label))
    print(f'OneHotCrossEntropyLoss with smooth label: {v}')

    # LabelSmoothCrossEntropyLoss test
    smooth_crit = LabelSmoothCrossEntropyLoss(smoothing=0.3)
    v = smooth_crit(Variable(predict),
	     Variable(label))
    print(f'LabelSmoothCrossEntropyLoss (1D labels but smooth internally): {v}')
