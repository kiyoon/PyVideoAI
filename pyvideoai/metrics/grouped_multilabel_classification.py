# https://mmuratarat.github.io/2020-01-25/multilabel_classification_metrics
from __future__ import annotations
import torch
from .metric import Metric

EPS = 1e-6


class ClipGroupedIOUAccuracyMetric(Metric):
    """
    Multi-label accuracy defined as the proportion of the predicted correct labels
    to the total number (predicted and actual) of labels for that instance.
    Defined in (Sorower, 2010).

    Grouping used for head/tail analysis.
    If a sample includes all head labels, it is counted in head group.
    If a sample includes all tail labels, it is counted in tail group.
    If a sample includes head and tail labels, we IGNORE them.

    """
    def __init__(self, class_groups: list[list[int]], group_names: list[str], activation = 'sigmoid', threshold = 0.5, **kwargs):
        self.num_groups = len(class_groups)
        self.class_groups = class_groups
        self.group_names = group_names
        assert len(class_groups) == len(group_names), f'Length of class_groups ({len(class_groups)}) and group_names ({len(group_names)}) does not match.'

        super().__init__(activation=activation, **kwargs)

        self.label_to_group_idx = {}
        for idx, class_group in enumerate(class_groups):
            for label_in_group in class_group:
                assert label_in_group not in self.label_to_group_idx.keys(), f'Label {label_in_group} seems to be in mutliple groups {self.label_to_group_idx[label_in_group]} and {idx}.'
                self.label_to_group_idx[label_in_group] = idx

        self.threshold = threshold

    def clean_data(self):
        super().clean_data()
        self.accuracy_per_clip = [[] for _ in range(self.num_groups)]
        self.last_calculated_metrics = (0.,) * self.num_groups


    def add_clip_predictions(self, video_ids, clip_predictions, labels: torch.Tensor):
        video_ids, clip_predictions, labels = super().add_clip_predictions(video_ids, clip_predictions, labels)
        if video_ids is None:
            return      # sometimes, after filtering the samples, there can be no samples to do anything.

        assert labels.dim() == 2, f'labels has to be a 2D tensor with ones and zeros but got {labels.dim()}-D.'

        clip_predictions = clip_predictions >= self.threshold

        for pred, label in zip(clip_predictions, labels):
            group_idx = None
            for class_idx, class_label in enumerate(label):
                assert class_label in [1., 0.], f'Label in IOU accuracy metric has to be ones and zeros but got {class_label}.'
                if class_label >= 1.-EPS:
                    if group_idx is None:
                        group_idx = self.label_to_group_idx[class_idx]
                    else:
                        if self.label_to_group_idx[class_idx] != group_idx:
                            group_idx = -1  # Doesn't belong to any group. Mixed group.
                            break

            if group_idx >= 0:
                accuracy_this_clip = torch.sum(torch.logical_and(pred, label)) / torch.sum(torch.logical_or(pred, label))
                self.accuracy_per_clip[group_idx].append(accuracy_this_clip.item())


    def calculate_metrics(self):
        self.last_calculated_metrics = tuple(sum(accuracy_per_clip) / len(accuracy_per_clip) if len(accuracy_per_clip) > 0 else 0. for accuracy_per_clip in self.accuracy_per_clip)


    def types_of_metrics(self):
        return (float, ) * self.num_groups


    def tensorboard_tags(self):
        return tuple(f'IOU {name} accuracy' for name in self.group_names)


    def get_csv_fieldnames(self):
        return tuple(f'{self.split}_iou_{name}_accuracy' for name in self.group_names)


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

        messages = [f'{prefix}iou{name}acc: {value:.4f}' for name, value in zip(self.group_names, self.last_calculated_metrics)]
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
            return tuple(f'Training IOU {name} accuracy' for name in self.group_names)
        elif self.split == 'val':
            return tuple(f'Validation IOU {name} accuracy' for name in self.group_names)
        elif self.split == 'multicropval':
            return tuple(f'Multicrop validation IOU {name} accuracy' for name in self.group_names)
        else:
            return tuple(f'{self.split} IOU {name} accuracy' for name in self.group_names)


    def plot_file_basenames(self):
        """
        Return:
            either tuple or a single str
        """
        # output plot file names will be e.g.) accuracy.png/pdf, accuracy_top5.png/pdf, ...
        return tuple(f'group_iou_{name}_accuracy' for name in self.group_names)


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
        return sum([len(accuracy_per_clip) for accuracy_per_clip in self.accuracy_per_clip])


class ClipGroupedF1MeasureMetric(Metric):
    """
    F1-measure (harmonic mean of precision and recall):
    mean(2*1(y_true & y_pred) / (1(y_true) + 1(y_pred)))
    where 1(x) = number of positive classes.

    Defined in (Sorower, 2010).

    Grouping used for head/tail analysis.
    If a sample includes all head labels, it is counted in head group.
    If a sample includes all tail labels, it is counted in tail group.
    If a sample includes head and tail labels, we IGNORE them.
    """
    def __init__(self, class_groups: list[list[int]], group_names: list[str], activation = 'sigmoid', threshold = 0.5, **kwargs):
        self.num_groups = len(class_groups)
        self.class_groups = class_groups
        self.group_names = group_names
        assert len(class_groups) == len(group_names), f'Length of class_groups ({len(class_groups)}) and group_names ({len(group_names)}) does not match.'

        super().__init__(activation=activation, **kwargs)

        self.label_to_group_idx = {}
        for idx, class_group in enumerate(class_groups):
            for label_in_group in class_group:
                assert label_in_group not in self.label_to_group_idx.keys(), f'Label {label_in_group} seems to be in mutliple groups {self.label_to_group_idx[label_in_group]} and {idx}.'
                self.label_to_group_idx[label_in_group] = idx

        self.threshold = threshold


    def clean_data(self):
        super().clean_data()
        self.f1_per_clip = [[] for _ in range(self.num_groups)]
        self.last_calculated_metrics = (0.,) * self.num_groups


    def add_clip_predictions(self, video_ids, clip_predictions, labels: torch.Tensor):
        video_ids, clip_predictions, labels = super().add_clip_predictions(video_ids, clip_predictions, labels)
        if video_ids is None:
            return      # sometimes, after filtering the samples, there can be no samples to do anything.

        assert labels.dim() == 2, f'labels has to be a 2D tensor with ones and zeros but got {labels.dim()}-D.'

        clip_predictions = clip_predictions >= self.threshold

        for pred, label in zip(clip_predictions, labels):
            group_idx = None
            for class_idx, class_label in enumerate(label):
                assert class_label in [1., 0.], f'Label in F1 measure metric has to be ones and zeros but got {class_label}.'
                if class_label >= 1.-EPS:
                    if group_idx is None:
                        group_idx = self.label_to_group_idx[class_idx]
                    else:
                        if self.label_to_group_idx[class_idx] != group_idx:
                            group_idx = -1  # Doesn't belong to any group. Mixed group.
                            break

            if group_idx >= 0:
                f1_this_clip = 2 * torch.sum(torch.logical_and(pred, label)) / (torch.sum(pred) + torch.sum(label))
                self.f1_per_clip[group_idx].append(f1_this_clip.item())


    def calculate_metrics(self):
        self.last_calculated_metrics = tuple(sum(f1_per_clip) / len(f1_per_clip) if len(f1_per_clip) > 0 else 0. for f1_per_clip in self.f1_per_clip)


    def types_of_metrics(self):
        return (float, ) * self.num_groups


    def tensorboard_tags(self):
        return tuple(f'F1 {name}' for name in self.group_names)


    def get_csv_fieldnames(self):
        return tuple(f'{self.split}_f1_{name}' for name in self.group_names)


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

        messages = [f'{prefix}f1{name}: {value:.4f}' for name, value in zip(self.group_names, self.last_calculated_metrics)]
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
            return tuple(f'Training F1 {name}' for name in self.group_names)
        elif self.split == 'val':
            return tuple(f'Validation F1 {name}' for name in self.group_names)
        elif self.split == 'multicropval':
            return tuple(f'Multicrop validation F1 {name}' for name in self.group_names)
        else:
            return tuple(f'{self.split} F1 {name}' for name in self.group_names)


    def plot_file_basenames(self):
        """
        Return:
            either tuple or a single str
        """
        # output plot file names will be e.g.) accuracy.png/pdf, accuracy_top5.png/pdf, ...
        return tuple(f'group_f1_{name}' for name in self.group_names)


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
        return sum([len(f1_per_clip) for f1_per_clip in self.f1_per_clip])
