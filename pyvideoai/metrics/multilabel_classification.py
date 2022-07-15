# https://mmuratarat.github.io/2020-01-25/multilabel_classification_metrics
import torch
from .metric import Metric



class ClipIOUAccuracyMetric(Metric):
    """
    Multi-label accuracy defined as the proportion of the predicted correct labels
    to the total number (predicted and actual) of labels for that instance.
    Defined in (Sorower, 2010).
    """
    def __init__(self, activation = 'sigmoid', threshold = 0.5, **kwargs):
        super().__init__(activation=activation, **kwargs)
        self.threshold = threshold

    def clean_data(self):
        super().clean_data()
        self.accuracy_per_clip = []


    def add_clip_predictions(self, video_ids, clip_predictions, labels: torch.Tensor):
        video_ids, clip_predictions, labels = super().add_clip_predictions(video_ids, clip_predictions, labels)
        if video_ids is None:
            return      # sometimes, after filtering the samples, there can be no samples to do anything.

        assert labels.dim() == 2, f'labels has to be a 2D tensor with ones and zeros but got {labels.dim()}-D.'

        clip_predictions = clip_predictions >= self.threshold

        for pred, label in zip(clip_predictions, labels):
            accuracy_this_clip = torch.sum(torch.logical_and(pred, label)) / torch.sum(torch.logical_or(pred, label))
            self.accuracy_per_clip.append(accuracy_this_clip.item())


    def calculate_metrics(self):
        self.last_calculated_metrics = sum(self.accuracy_per_clip) / len(self) if len(self) > 0 else 0.


    def types_of_metrics(self):
        return float


    def tensorboard_tags(self):
        return 'IOU accuracy'


    def get_csv_fieldnames(self):
        return f'{self.split}_iou_accuracy'


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

        message = f'{prefix}iouacc: {self.last_calculated_metrics:.4f}'
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
            return 'Training IOU accuracy'
        elif self.split == 'val':
            return 'Validation IOU accuracy'
        elif self.split == 'multicropval':
            return 'Multicrop validation IOU accuracy'
        else:
            return f'{self.split} IOU accuracy'


    def plot_file_basenames(self):
        """
        Return:
            either tuple or a single str
        """
        # output plot file names will be e.g.) accuracy.png/pdf, accuracy_top5.png/pdf, ...
        return 'iou_accuracy'


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
        return len(self.accuracy_per_clip)


class ClipF1MeasureMetric(Metric):
    """
    F1-measure (harmonic mean of precision and recall):
    mean(2*1(y_true & y_pred) / (1(y_true) + 1(y_pred)))
    where 1(x) = number of positive classes.

    Defined in (Sorower, 2010).
    """
    def __init__(self, activation = 'sigmoid', threshold = 0.5, **kwargs):
        super().__init__(activation=activation, **kwargs)
        self.threshold = threshold

    def clean_data(self):
        super().clean_data()
        self.f1_per_clip = []


    def add_clip_predictions(self, video_ids, clip_predictions, labels: torch.Tensor):
        video_ids, clip_predictions, labels = super().add_clip_predictions(video_ids, clip_predictions, labels)
        if video_ids is None:
            return      # sometimes, after filtering the samples, there can be no samples to do anything.

        assert labels.dim() == 2, f'labels has to be a 2D tensor with ones and zeros but got {labels.dim()}-D.'

        clip_predictions = clip_predictions >= self.threshold

        for pred, label in zip(clip_predictions, labels):
            f1_this_clip = 2 * torch.sum(torch.logical_and(pred, label)) / (torch.sum(pred) + torch.sum(label))
            self.f1_per_clip.append(f1_this_clip.item())


    def calculate_metrics(self):
        self.last_calculated_metrics = sum(self.f1_per_clip) / len(self) if len(self) > 0 else 0.


    def types_of_metrics(self):
        return float


    def tensorboard_tags(self):
        return 'F1-Measure'


    def get_csv_fieldnames(self):
        return f'{self.split}_f1_measure'


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

        message = f'{prefix}f1: {self.last_calculated_metrics:.4f}'
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
            return 'Training F1-measure'
        elif self.split == 'val':
            return 'Validation F1-measure'
        elif self.split == 'multicropval':
            return 'Multicrop validation F1-measure'
        else:
            return f'{self.split} F1-measure'


    def plot_file_basenames(self):
        """
        Return:
            either tuple or a single str
        """
        # output plot file names will be e.g.) accuracy.png/pdf, accuracy_top5.png/pdf, ...
        return 'f1-measure'


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
        return len(self.f1_per_clip)
