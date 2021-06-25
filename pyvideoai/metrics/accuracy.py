import torch
import numpy as np
from .metric import ClipMetric

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k using PyTorch"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size).item())
    return res


class ClipAccuracyMetric(ClipMetric):
    def __init__(self, topk=(1,)):
        super().__init__()
        if isinstance(topk, tuple):
            self.topk = topk
        elif isinstance(topk, int):
            self.topk = (topk,)
        else:
            raise ValueError(f'topk {topk} not recognised. It must be a tuple or integer.')

    def calculate_metrics(self):
        video_predictions, video_labels, _ = self.get_predictions_torch()
        accuracies = accuracy(video_predictions, video_labels, self.topk)
        return tuple(accuracies)


    def types_of_metrics(self):
        return (float,) * len(self.topk)


    def tensorboard_tags(self):
        return tuple([f'clip_accuracy_top{topk}' for topk in self.topk])


    def get_csv_fieldnames(self, split):
        return tuple([f'{split}_acc_top{topk}' if topk != 1 else f'{split}_acc' for topk in self.topk])


    def logging_str_iter(self, split, value):
        """
        Returning None will skip logging this metric
        """
        if split == 'train':
            prefix = ''
        else:
            prefix = f'{split}_'

        return tuple([f'{prefix}acc: {value:.4f}' for topk in self.topk])


    def logging_str_epoch(self, split, value):
        """
        Returning None will skip logging this metric
        """
        return self.logging_str_iter(split, value)
    
    @staticmethod
    def is_better(value_1, value_2):
        """Metric comparison function

        params:
            two values of the metric

        return:
            True if value_1 is better. False if value_2 is better or they're equal.
        """
        return value_1 > value_2 


class VideoAccuracyMetric(AverageMetric):
    def __init__(self, topk=(1,)):
        super().__init__()
        if isinstance(topk, tuple):
            self.topk = topk
        elif isinstance(topk, int):
            self.topk = (topk,)
        else:
            raise ValueError(f'topk {topk} not recognised. It must be a tuple or integer.')

    def calculate_metrics(self):
        video_predictions, video_labels, _ = self.get_predictions_torch()
        accuracies = accuracy(video_predictions, video_labels, self.topk)
        return tuple(accuracies)


    def types_of_metrics(self):
        return (float,) * len(self.topk)


    def tensorboard_tags(self):
        return tuple([f'video_accuracy_top{topk}' for topk in self.topk])


    def get_csv_fieldnames(self, split):
        return tuple([f'{split}_vid_acc_top{topk}' if topk != 1 else f'{split}_vid_acc' for topk in self.topk])


    def logging_str_iter(self, split, value):
        """
        Returning None will skip logging this metric
        """
        return None


    def logging_str_epoch(self, split, value):
        """
        Returning None will skip logging this metric
        """
        if split == 'train':
            prefix = ''
        else:
            prefix = f'{split}_'

        return tuple([f'{prefix}vid_acc: {value:.4f}' for topk in self.topk])
    
    @staticmethod
    def is_better(value_1, value_2):
        """Metric comparison function

        params:
            two values of the metric

        return:
            True if value_1 is better. False if value_2 is better or they're equal.
        """
        return value_1 > value_2 

