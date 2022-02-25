import torch
import numpy as np
from .metric import ClipMetric, AverageMetric

from .AP import mAP
from sklearn.metrics import average_precision_score

class Clip_mAPMetric(ClipMetric):
    def __init__(self, activation='sigmoid', video_id_to_label: dict[np.array] = None, backend='CATER'):
        super().__init__(activation=activation, video_id_to_label = video_id_to_label)

        assert backend in ['CATER', 'sklearn'], f'Backend not identified: {backend}'
        self.backend = backend


    def calculate_metrics(self):
        if len(self) > 0:
            video_predictions, video_labels, _ = self.get_predictions_numpy()
            if self.backend == 'CATER':
                self.last_calculated_metrics = mAP(video_labels, video_predictions)
            elif self.backend == 'sklearn':
                self.last_calculated_metrics = average_precision_score(video_labels, video_predictions, average='macro')
            else:
                raise ValueError(f'self.backend not recognised: {self.backend}')
        else:
            self.last_calculated_metrics = 0.


    def types_of_metrics(self):
        return float


    def tensorboard_tags(self):
        return 'Clip mAP'


    def get_csv_fieldnames(self):
        return f'{self.split}_mAP'


    def logging_msg_iter(self):
        """
        Return:
            None to skip logging this metric
            or a single str that combines all self.last_calculated_metrics 
        """
        return None


    def logging_msg_epoch(self):
        """
        Return:
            None to skip logging this metric
            or a single str that combines all self.last_calculated_metrics 
        """
        if self.split == 'train':
            prefix = ''
        else:
            prefix = f'{self.split}_'

        return f'{prefix}mAP: {self.last_calculated_metrics:.4f}'


    def plot_legend_labels(self):
        """
        Return:
            either tuple or a single str 
        """
        if self.split == 'train':
            return 'Training mAP'
        elif self.split == 'val':
            return 'Validation mAP'
        elif self.split == 'multicropval':
            return 'Multicrop validation clip mAP'
        else:
            return f'{self.split} clip mAP'


    def plot_file_basenames(self):
        """
        Return:
            either tuple or a single str 
        """
        # output plot file names will be e.g.) accuracy.png/pdf, accuracy_top5.png/pdf, ...
        return 'mAP'

    
    @staticmethod
    def is_better(value_1, value_2):
        """Metric comparison function

        params:
            two values of the metric

        return:
            True if value_1 is better. False if value_2 is better or they're equal.
        """
        return value_1 > value_2 


class Video_mAPMetric(AverageMetric):
    def __init__(self, activation='sigmoid', video_id_to_label: dict[np.array] = None, backend='CATER'):
        super().__init__(activation=activation, video_id_to_label = video_id_to_label)

        assert backend in ['CATER', 'sklearn'], f'Backend not identified: {backend}'
        self.backend = backend


    def calculate_metrics(self):
        if len(self) > 0:
            video_predictions, video_labels, _ = self.get_predictions_numpy()
            if self.backend == 'CATER':
                self.last_calculated_metrics = mAP(video_labels, video_predictions)
            elif self.backend == 'sklearn':
                self.last_calculated_metrics = average_precision_score(video_labels, video_predictions, average='macro')
            else:
                raise ValueError(f'self.backend not recognised: {self.backend}')
        else:
            self.last_calculated_metrics = 0.


    def types_of_metrics(self):
        return float


    def tensorboard_tags(self):
        return 'Video mAP'


    def get_csv_fieldnames(self):
        return f'{self.split}_vid_mAP'


    def logging_msg_iter(self):
        """
        Return:
            None to skip logging this metric
            or a single str that combines all self.last_calculated_metrics 
        """
        return None


    def logging_msg_epoch(self):
        """
        Return:
            None to skip logging this metric
            or a single str that combines all self.last_calculated_metrics 
        """
        if self.split == 'train':
            prefix = ''
        else:
            prefix = f'{self.split}_'

        return f'{prefix}vid_mAP: {self.last_calculated_metrics:.4f}'


    def plot_legend_labels(self):
        """
        Return:
            either tuple or a single str 
        """
        if self.split == 'train':
            return 'Training video mAP'
        elif self.split == 'val':
            return 'Validation video mAP'
        elif self.split == 'multicropval':
            return 'Multicrop validation video mAP'
        else:
            return f'{self.split} video mAP'


    def plot_file_basenames(self):
        """
        Return:
            either tuple or a single str 
        """
        # output plot file names will be e.g.) accuracy.png/pdf, accuracy_top5.png/pdf, ...
        return 'mAP'

    
    @staticmethod
    def is_better(value_1, value_2):
        """Metric comparison function

        params:
            two values of the metric

        return:
            True if value_1 is better. False if value_2 is better or they're equal.
        """
        return value_1 > value_2 
