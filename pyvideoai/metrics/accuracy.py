import torch
from .metric import Metric, AverageMetric
from ..utils.misc import check_single_label_target_and_convert_to_numpy

import logging
logger = logging.getLogger(__name__)

def accuracy(output:torch.Tensor, target:torch.Tensor, topk=(1,)):
    """Computes the precision@k for the specified values of k using PyTorch.
    Kiyoon edit: target possible to be either 1D tensor or one-hot/labelsmooth 2D tensor.

    """
    assert target.dim() in [1,2], f'target has to be 1D or 2D tensor but got {target.dim()}-D.'

    maxk = max(topk)
    if target.dim() == 2:
        # Convert 2D one-hot label to 1D label
        target = target.argmax(dim=1)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
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
    def __init__(self, topk=(1,), verify_singlelabel: bool = False, **kwargs):
        if isinstance(topk, tuple):
            self.topk = topk
        elif isinstance(topk, int):
            self.topk = (topk,)
        else:
            raise ValueError(f'topk {topk} not recognised. It must be a tuple or integer.')

        self.verify_singlelabel = verify_singlelabel
        if self.verify_singlelabel:
            logger.warning('verify_singlelabel is ON when calculating accuracy metric. Only for debugging use and should be turned off for speed. It will copy GPU tensors to CPU and check every elements to see if it is the right format.')

        super().__init__(**kwargs)


    def clean_data(self):
        super().clean_data()
        self.num_seen_samples = 0
        self.num_correct_preds = [0] * len(self.topk)
        self.last_calculated_metrics = (0.,) * len(self.topk)


    def add_clip_predictions(self, video_ids, clip_predictions, labels):
        video_ids, clip_predictions, labels = super().add_clip_predictions(video_ids, clip_predictions, labels)
        if video_ids is None:
            return      # sometimes, after filtering the samples, there can be no samples to do anything.

        if self.verify_singlelabel:
            check_single_label_target_and_convert_to_numpy(labels)

        maxk = max(self.topk)
        assert labels.dim() in [1,2], f'labels has to be 1D or 2D tensor but got {labels.dim()}-D.'
        if labels.dim() == 2:
            # Convert 2D one-hot label to 1D label
            labels = labels.argmax(dim=1)
        batch_size = labels.size(0)

        _, pred = clip_predictions.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))

        for i, k in enumerate(self.topk):
            batch_correct = correct[:k].reshape(-1).int().sum(0)
            self.num_correct_preds[i] += batch_correct
        self.num_seen_samples += batch_size


    def calculate_metrics(self):
        self.last_calculated_metrics = tuple([(num_correct / self.num_seen_samples).item() if len(self) > 0 else 0. for num_correct in self.num_correct_preds])


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
            prefix = f'{self.split}_'

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
            return tuple([f'{self.split} accuracy top{topk}' if topk != 1 else f'{self.split} clip accuracy' for topk in self.topk])


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
    def __init__(self, topk=(1,), **kwargs):
        if isinstance(topk, tuple):
            self.topk = topk
        elif isinstance(topk, int):
            self.topk = (topk,)
        else:
            raise ValueError(f'topk {topk} not recognised. It must be a tuple or integer.')

        super().__init__(**kwargs)

    def clean_data(self):
        super().clean_data()
        self.last_calculated_metrics = (0.,) * len(self.topk)

    def calculate_metrics(self):
        if len(self) > 0:
            video_predictions, video_labels, _ = self.get_predictions_torch()
            self.last_calculated_metrics = accuracy(video_predictions, video_labels, self.topk)
        else:
            self.last_calculated_metrics = (0.,) * len(self.topk)


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
            return tuple([f'{self.split} video accuracy top{topk}' if topk != 1 else f'{self.split} video accuracy' for topk in self.topk])


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
