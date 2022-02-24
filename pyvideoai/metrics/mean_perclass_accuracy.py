import torch
import numpy as np
from .metric import Metric, AverageMetric



class ClipMeanPerclassAccuracyMetric(Metric):
    """Don't need activation softmax for clip accuracy calculation.
    """
    def __init__(self, activation=None):
        super().__init__(activation=activation)


    def clean_data(self):
        super().clean_data()
        self.num_seen_samples = None
        self.num_true_positives = None 



    def add_clip_predictions(self, video_ids, clip_predictions, labels):
        super().add_clip_predictions(video_ids, clip_predictions, labels)

        assert labels.dim() in [1,2], f'target has to be 1D or 2D tensor but got {target.dim()}-D.'
        if labels.dim() == 2:
            # Convert 2D one-hot label to 1D label
            labels = labels.argmax(dim=1)

        if self.num_seen_samples is None:
            self.num_seen_samples = [0] * self.num_classes
            self.num_true_positives = [0] * self.num_classes

        pred_labels = torch.argmax(clip_predictions, dim=1)
        for pred, label in zip(pred_labels, labels):
            self.num_seen_samples[label] += 1
            if pred == label:
                self.num_true_positives[label] += 1


    def calculate_metrics(self):
        filtered_perclass_accuracies = [tp / sample_count for tp, sample_count in zip(self.num_true_positives, self.num_seen_samples) if sample_count > 0]
        self.last_calculated_metrics = sum(filtered_perclass_accuracies) / len(filtered_perclass_accuracies)

    def types_of_metrics(self):
        return float


    def tensorboard_tags(self):
        return 'Clip mean per-class accuracy'


    def get_csv_fieldnames(self):
        return f'{self.split}_meanperclassacc'


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

        message = f'{prefix}meanperclassacc: {self.last_calculated_metrics:.4f}'
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
            return 'Training average per-class accuracy'
        elif self.split == 'val':
            return 'Validation average per-class accuracy'
        elif self.split == 'multicropval':
            return 'Multicrop validation average per-class accuracy'
        else:
            return f'{self.split} average per-class accuracy'


    def plot_file_basenames(self):
        """
        Return:
            either tuple or a single str 
        """
        # output plot file names will be e.g.) accuracy.png/pdf, accuracy_top5.png/pdf, ...
        return 'mean_perclass_accuracy'

    
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

