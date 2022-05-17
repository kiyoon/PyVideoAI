from __future__ import annotations
import torch
from .metric import Metric



class ClipGroupedClassAccuracyMetric(Metric):
    """Don't need activation softmax for clip accuracy calculation.
    Compute accuracy for each group of classes. For example, head and tail classes.

    """
    def __init__(self, class_groups: list[list[int]], group_names: list[str], **kwargs):
        """
        Params:
            class_groups: list of list of class indices. For example, if 0,2 is a head class group and 1,3 is a tail group, [[0,2], [1,3]].
            group_names: name of each group. For example, ['head', 'tail']
        """
        self.num_groups = len(class_groups)
        self.class_groups = class_groups
        self.group_names = group_names
        assert len(class_groups) == len(group_names), f'Length of class_groups ({len(class_groups)}) and group_names ({len(group_names)}) does not match.'

        super().__init__(**kwargs)

        self.label_to_group_idx = {}
        for idx, class_group in enumerate(class_groups):
            for label_in_group in class_group:
                assert label_in_group not in self.label_to_group_idx.keys(), f'Label {label_in_group} seems to be in mutliple groups {self.label_to_group_idx[label_in_group]} and {idx}.'
                self.label_to_group_idx[label_in_group] = idx



    def clean_data(self):
        """
        Note that this doesn't change the group definitions.
        """
        super().clean_data()
        self.num_seen_samples = [0] * self.num_groups
        self.num_true_positives = [0] * self.num_groups
        self.last_calculated_metrics = (0.,) * self.num_groups


    def add_clip_predictions(self, video_ids, clip_predictions, labels):
        video_ids, clip_predictions, labels = super().add_clip_predictions(video_ids, clip_predictions, labels)
        if video_ids is None:
            return      # sometimes, after filtering the samples, there can be no samples to do anything.

        assert labels.dim() in [1,2], f'labels has to be 1D or 2D tensor but got {labels.dim()}-D.'
        if labels.dim() == 2:
            # Convert 2D one-hot label to 1D label
            labels = labels.argmax(dim=1)

        pred_labels = torch.argmax(clip_predictions, dim=1)
        for pred, label in zip(pred_labels, labels):
            group_idx = self.label_to_group_idx[label.item()]
            self.num_seen_samples[group_idx] += 1
            if pred == label:
                self.num_true_positives[group_idx] += 1


    def calculate_metrics(self):
        self.last_calculated_metrics = tuple(tp / sample_count if sample_count > 0 else 0. for tp, sample_count in zip(self.num_true_positives, self.num_seen_samples))


    def types_of_metrics(self):
        return (float, ) * self.num_groups


    def tensorboard_tags(self):
        return tuple(f'{name} accuracy' for name in self.group_names)


    def get_csv_fieldnames(self):
        return tuple(f'{self.split}_{name}_accuracy' for name in self.group_names)


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

        messages = [f'{prefix}{name}acc: {value:.4f}' for name, value in zip(self.group_names, self.last_calculated_metrics)]
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
            return tuple(f'Training {name} accuracy' for name in self.group_names)
        elif self.split == 'val':
            return tuple(f'Validation {name} accuracy' for name in self.group_names)
        elif self.split == 'multicropval':
            return tuple(f'Multicrop validation {name} accuracy' for name in self.group_names)
        else:
            return tuple(f'{self.split} {name} accuracy' for name in self.group_names)


    def plot_file_basenames(self):
        """
        Return:
            either tuple or a single str
        """
        # output plot file names will be e.g.) accuracy.png/pdf, accuracy_top5.png/pdf, ...
        return tuple(f'group_{name}_accuracy' for name in self.group_names)


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
        return sum(self.num_seen_samples)
