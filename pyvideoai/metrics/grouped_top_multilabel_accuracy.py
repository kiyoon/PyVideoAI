from __future__ import annotations
import torch
from .metric import Metric

EPS = 1e-6


class ClipGroupedTop1MultilabelAccuracyMetric(Metric):
    """
    Used for head/tail analysis.
    Count as true positive when the top-1 prediction is in one of the head/tail ground truth labels.

    Don't need activation softmax for clip accuracy calculation.
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
        super().clean_data()
        self.num_seen_samples = [0] * self.num_groups
        self.num_true_positives = [0] * self.num_groups
        self.last_calculated_metrics = (0.,) * self.num_groups


    def add_clip_predictions(self, video_ids, clip_predictions, labels):
        video_ids, clip_predictions, labels = super().add_clip_predictions(video_ids, clip_predictions, labels)
        if video_ids is None:
            return      # sometimes, after filtering the samples, there can be no samples to do anything.

        assert labels.dim() == 2, f'labels has to be a 2D tensor with ones and zeros but got {labels.dim()}-D.'

        pred_labels = torch.argmax(clip_predictions, dim=1)
        for pred, label in zip(pred_labels, labels):
            is_in_group = [False] * self.num_groups
            for class_idx, class_label in enumerate(label):
                assert class_label in [1., 0.], f'Label in Top1 Multilabel Accuracy metric has to be ones and zeros but got {class_label}.'
                if class_label >= 1.-EPS:
                    group_idx = self.label_to_group_idx[class_idx]

                    # in order not to count the sample twice just because there are multiple head/tail ground truth.
                    if not is_in_group[group_idx]:
                        is_in_group[group_idx] = True
                        self.num_seen_samples[group_idx] += 1

                    if pred == class_idx:
                        self.num_true_positives[group_idx] += 1


    def calculate_metrics(self):
        self.last_calculated_metrics = tuple(tp / sample_count if sample_count > 0 else 0. for tp, sample_count in zip(self.num_true_positives, self.num_seen_samples))


    def types_of_metrics(self):
        return (float, ) * self.num_groups


    def tensorboard_tags(self):
        return tuple(f'Top1 multilabel {name} accuracy' for name in self.group_names)


    def get_csv_fieldnames(self):
        return tuple(f'{self.split}_top1_multilabel_{name}_accuracy' for name in self.group_names)


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

        messages = [f'{prefix}top1multi{name}acc: {value:.4f}' for name, value in zip(self.group_names, self.last_calculated_metrics)]
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
            return tuple(f'Training top1 multilabel {name} accuracy' for name in self.group_names)
        elif self.split == 'val':
            return tuple(f'Validation top1 multilabel {name} accuracy' for name in self.group_names)
        elif self.split == 'multicropval':
            return tuple(f'Multicrop validation top1 multilabel {name} accuracy' for name in self.group_names)
        else:
            return tuple(f'{self.split} top1 multilabel {name} accuracy' for name in self.group_names)


    def plot_file_basenames(self):
        """
        Return:
            either tuple or a single str
        """
        # output plot file names will be e.g.) accuracy.png/pdf, accuracy_top5.png/pdf, ...
        return tuple(f'group_top1_multilabel_{name}_accuracy' for name in self.group_names)


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


class ClipGroupedTopkMultilabelAccuracyMetric(Metric):
    """
    Used for head/tail analysis.
    Count as true positive when the top-k prediction is in one of the k head/tail ground truth labels.

    Don't need activation softmax for clip accuracy calculation.
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
        super().clean_data()
        self.num_seen_samples = [0] * self.num_groups
        self.num_true_positives = [0] * self.num_groups
        self.last_calculated_metrics = (0.,) * self.num_groups


    def add_clip_predictions(self, video_ids, clip_predictions, labels):
        video_ids, clip_predictions, labels = super().add_clip_predictions(video_ids, clip_predictions, labels)
        if video_ids is None:
            return      # sometimes, after filtering the samples, there can be no samples to do anything.

        assert labels.dim() == 2, f'labels has to be a 2D tensor with ones and zeros but got {labels.dim()}-D.'

        for pred, label in zip(clip_predictions, labels):
            labels_cur_sample = torch.where(label>=1.-EPS)[0]
            num_labels = labels_cur_sample.size(0)
            num_nonlabels = torch.where(label<0.+EPS)[0].size(0)
            if num_labels + num_nonlabels != self.num_classes:
                raise ValueError(('Label in Top-K Multilabel Accuracy metric has to be ones and zeros but got something else.\n'
                f'Num_ones = {num_labels}, num_zeros = {num_nonlabels}, num_classes = {self.num_classes}'))

            # Add the sample to seen, per group.
            # If one of the labels is in the group, count in the group.
            is_in_group = [False] * self.num_groups
            for class_label in labels_cur_sample:
                group_idx = self.label_to_group_idx[class_label.item()]
                if not is_in_group[group_idx]:
                    is_in_group[group_idx] = True
                    self.num_seen_samples[group_idx] += 1

            # Add true positives per group
            num_tp_diff_per_group = [0] * self.num_groups
            _, top_classes = torch.topk(pred, num_labels)
            for top_class in top_classes:
                if top_class in labels_cur_sample:
                    group_idx = self.label_to_group_idx[top_class.item()]
                    num_tp_diff_per_group[group_idx] = 1    # never increase more than 1 per sample.

            for group_idx, num_tp_diff in enumerate(num_tp_diff_per_group):
                self.num_true_positives[group_idx] += num_tp_diff



    def calculate_metrics(self):
        self.last_calculated_metrics = tuple(tp / sample_count if sample_count > 0 else 0. for tp, sample_count in zip(self.num_true_positives, self.num_seen_samples))


    def types_of_metrics(self):
        return (float, ) * self.num_groups


    def tensorboard_tags(self):
        return tuple(f'Top-K multilabel {name} accuracy' for name in self.group_names)


    def get_csv_fieldnames(self):
        return tuple(f'{self.split}_topk_multilabel_{name}_accuracy' for name in self.group_names)


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

        messages = [f'{prefix}topkmulti{name}acc: {value:.4f}' for name, value in zip(self.group_names, self.last_calculated_metrics)]
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
            return tuple(f'Training top-k multilabel {name} accuracy' for name in self.group_names)
        elif self.split == 'val':
            return tuple(f'Validation top-k multilabel {name} accuracy' for name in self.group_names)
        elif self.split == 'multicropval':
            return tuple(f'Multicrop validation top-k multilabel {name} accuracy' for name in self.group_names)
        else:
            return tuple(f'{self.split} top-k multilabel {name} accuracy' for name in self.group_names)


    def plot_file_basenames(self):
        """
        Return:
            either tuple or a single str
        """
        # output plot file names will be e.g.) accuracy.png/pdf, accuracy_top5.png/pdf, ...
        return tuple(f'group_topk_multilabel_{name}_accuracy' for name in self.group_names)


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
