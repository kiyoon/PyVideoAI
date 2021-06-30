import torch
import numpy as np
from .metric import Metric, AverageMetric

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



class ClipAccuracyMetric(Metric):
    """Don't need activation softmax for clip accuracy calculation.
    """
    def __init__(self, topk=(1,), activation=None):
        if isinstance(topk, tuple):
            self.topk = topk
        elif isinstance(topk, int):
            self.topk = (topk,)
        else:
            raise ValueError(f'topk {topk} not recognised. It must be a tuple or integer.')

        super().__init__(activation=activation)


    def clean_data(self):
        super().clean_data()
        self.num_seen_samples = 0
        self.num_correct_preds = [0] * len(self.topk)
        self.last_calculated_metrics = (0.,) * len(self.topk)


    def add_clip_predictions(self, video_ids, clip_predictions, labels):
        super().add_clip_predictions(video_ids, clip_predictions, labels)

        maxk = max(self.topk)
        batch_size = labels.size(0)

        _, pred = clip_predictions.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))

        res = []
        for i, k in enumerate(self.topk):
            batch_correct = correct[:k].reshape(-1).int().sum(0)
            self.num_correct_preds[i] += batch_correct
        self.num_seen_samples += batch_size 


    def calculate_metrics(self):
        self.last_calculated_metrics = tuple([num_correct / self.num_seen_samples for num_correct in self.num_correct_preds])


    def types_of_metrics(self):
        return (float,) * len(self.topk)


    def tensorboard_tags(self):
        return tuple([f'Clip accuracy top{topk}' if topk != 1 else 'Clip accuracy' for topk in self.topk])


    def get_csv_fieldnames(self):
        return tuple([f'{self.split}_acc_top{topk}' if topk != 1 else f'{self.split}_acc' for topk in self.topk])


    def logging_msg_iter(self):
        """
        Return:
            None to skip logging this metric
            or a single str that combines all self.last_calculated_metrics 
        """
        if self.split == 'train':
            prefix = ''
        else:
            split = self.split.replace('multicrop', '')  # don't print multicrop for one clip evaluation
            prefix = f'{split}_'

        messages = [f'{prefix}acc: {value:.4f}' if topk == 1 else f'{prefix}acc_top{topk}: {value:.4f}' for topk, value in zip(self.topk, self.last_calculated_metrics)]
        message = ' - '.join(messages)
        return message


    def logging_msg_epoch(self):
        """
        Return:
            None to skip logging this metric
            or a single str that combines all self.last_calculated_metrics 
        """
        return self.logging_msg_iter()


    def plot_legend_labels(self):
        """
        Return:
            either tuple or a single str 
        """
        if self.split == 'train':
            return tuple([f'Training accuracy top{topk}' if topk != 1 else 'Training accuracy' for topk in self.topk])
        elif self.split == 'val':
            return tuple([f'Validation accuracy top{topk}' if topk != 1 else 'Validation accuracy' for topk in self.topk])
        elif self.split == 'multicropval':
            return tuple([f'Multicrop validation clip accuracy top{topk}' if topk != 1 else 'Multicrop validation clip accuracy' for topk in self.topk])
        else:
            raise ValueError(f'Unknown split: {self.split}')


    def plot_file_basenames(self):
        """
        Return:
            either tuple or a single str 
        """
        # output plot file names will be e.g.) accuracy.png/pdf, accuracy_top5.png/pdf, ...
        return tuple([f'accuracy_top{topk}' if topk != 1 else 'accuracy' for topk in self.topk])

    
    @staticmethod
    def is_better(value_1, value_2):
        """Metric comparison function

        params:
            two values of the metric

        return:
            True if value_1 is better. False if value_2 is better or they're equal.
        """
        return value_1 > value_2 

    def __len__(self):
        return self.num_seen_samples

class VideoAccuracyMetric(AverageMetric):
    def __init__(self, topk=(1,), activation='softmax'):
        if isinstance(topk, tuple):
            self.topk = topk
        elif isinstance(topk, int):
            self.topk = (topk,)
        else:
            raise ValueError(f'topk {topk} not recognised. It must be a tuple or integer.')

        super().__init__(activation=activation)

    def clean_data(self):
        super().clean_data()
        self.last_calculated_metrics = (0.,) * len(self.topk)

    def calculate_metrics(self):
        video_predictions, video_labels, _ = self.get_predictions_torch()
        self.last_calculated_metrics = accuracy(video_predictions, video_labels, self.topk)


    def types_of_metrics(self):
        return (float,) * len(self.topk)


    def tensorboard_tags(self):
        return tuple([f'Video accuracy top{topk}' if topk != 1 else 'Video accuracy' for topk in self.topk])


    def get_csv_fieldnames(self):
        return tuple([f'{self.split}_vid_acc_top{topk}' if topk != 1 else f'{self.split}_vid_acc' for topk in self.topk])


    def logging_msg_iter(self):
        """
        Returning None will skip logging this metric
        """
        return None


    def logging_msg_epoch(self):
        """
        Returning None will skip logging this metric
        """
        if self.split == 'train':
            prefix = ''
        else:
            prefix = f'{self.split}_'

        messages = [f'{prefix}vid_acc: {value:.4f}' if topk == 1 else f'{prefix}vid_acc_top{topk}: {value:.4f}' for topk, value in zip(self.topk, self.last_calculated_metrics)]
        message = ' - '.join(messages)
        return message
    

    def plot_legend_labels(self):
        """
        Return:
            either tuple or a single str 
        """
        if self.split == 'train':
            return tuple([f'Training video accuracy top{topk}' if topk != 1 else 'Training video accuracy' for topk in self.topk])
        elif self.split == 'val':
            return tuple([f'Validation video accuracy top{topk}' if topk != 1 else 'Validation video accuracy' for topk in self.topk])
        elif self.split == 'multicropval':
            return tuple([f'Multicrop validation video accuracy top{topk}' if topk != 1 else 'Multicrop validation video accuracy' for topk in self.topk])
        else:
            raise ValueError(f'Unknown split: {self.split}')


    def plot_file_basenames(self):
        """
        Return:
            either tuple or a single str 
        """
        # output plot file names will be e.g.) accuracy.png/pdf, accuracy_top5.png/pdf, ...
        # Plot within the same graph 
        return tuple([f'accuracy_top{topk}' if topk != 1 else 'accuracy' for topk in self.topk])


    @staticmethod
    def is_better(value_1, value_2):
        """Metric comparison function

        params:
            two values of the metric

        return:
            True if value_1 is better. False if value_2 is better or they're equal.
        """
        return value_1 > value_2 

