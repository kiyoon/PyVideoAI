from .metric import ClipMetric, AverageMetric

from .AP import mAP
from sklearn.metrics import average_precision_score

class Clip_mAPMetric(ClipMetric):
    def __init__(self, *, activation='sigmoid', backend='CATER', exclude_classes_less_sample_than=1, **kwargs):
        super().__init__(activation=activation, **kwargs)

        assert backend in ['CATER', 'sklearn'], f'Backend not identified: {backend}'
        assert exclude_classes_less_sample_than >= 1
        if exclude_classes_less_sample_than != 1:
            assert backend == 'CATER', 'Only CATER backend is supported with exclude_classes_less_sample_than > 1'

        self.exclude_classes_less_sample_than = exclude_classes_less_sample_than
        self.backend = backend


    def calculate_metrics(self):
        if len(self) > 0:
            video_predictions, video_labels, _ = self.get_predictions_numpy()
            if self.backend == 'CATER':
                self.last_calculated_metrics = mAP(video_labels, video_predictions, exclude_classes_less_sample_than = self.exclude_classes_less_sample_than)
            elif self.backend == 'sklearn':
                self.last_calculated_metrics = average_precision_score(video_labels, video_predictions, average='macro')
            else:
                raise ValueError(f'self.backend not recognised: {self.backend}')
        else:
            self.last_calculated_metrics = 0.


    def types_of_metrics(self):
        return float


    def tensorboard_tags(self):
        if self.exclude_classes_less_sample_than == 1:
            return 'Clip mAP'
        else:
            return 'Clip mAP {self.exclude_classes_less_sample_than}+'


    def get_csv_fieldnames(self):
        if self.exclude_classes_less_sample_than == 1:
            return f'{self.split}_mAP'
        else:
            return f'{self.split}_mAP{self.exclude_classes_less_sample_than}+'


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

        if self.exclude_classes_less_sample_than == 1:
            suffix = ''
        else:
            suffix = f'{self.exclude_classes_less_sample_than}+'

        return f'{prefix}mAP{suffix}: {self.last_calculated_metrics:.4f}'


    def plot_legend_labels(self):
        """
        Return:
            either tuple or a single str
        """
        if self.exclude_classes_less_sample_than == 1:
            suffix = ''
        else:
            suffix = f' ({self.exclude_classes_less_sample_than}+ samples)'

        if self.split == 'train':
            return f'Training mAP{suffix}'
        elif self.split == 'val':
            return f'Validation mAP{suffix}'
        elif self.split == 'multicropval':
            return f'Multicrop validation clip mAP{suffix}'
        else:
            return f'{self.split} clip mAP{suffix}'


    def plot_file_basenames(self):
        """
        Return:
            either tuple or a single str
        """
        # output plot file names will be e.g.) accuracy.png/pdf, accuracy_top5.png/pdf, ...
        if self.exclude_classes_less_sample_than == 1:
            suffix = ''
        else:
            suffix = f'_{self.exclude_classes_less_sample_than}+'

        return f'mAP{suffix}'


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
    def __init__(self, *, activation='sigmoid', backend='CATER', exclude_classes_less_sample_than = 1, **kwargs):
        super().__init__(activation=activation, **kwargs)

        assert exclude_classes_less_sample_than >= 1
        if exclude_classes_less_sample_than != 1:
            assert backend == 'CATER', 'Only CATER backend is supported with exclude_classes_less_sample_than > 1'

        self.exclude_classes_less_sample_than = exclude_classes_less_sample_than

        assert backend in ['CATER', 'sklearn'], f'Backend not identified: {backend}'
        self.backend = backend


    def calculate_metrics(self):
        if len(self) > 0:
            video_predictions, video_labels, _ = self.get_predictions_numpy()
            if self.backend == 'CATER':
                self.last_calculated_metrics = mAP(video_labels, video_predictions, exclude_classes_less_sample_than = self.exclude_classes_less_sample_than)
            elif self.backend == 'sklearn':
                self.last_calculated_metrics = average_precision_score(video_labels, video_predictions, average='macro')
            else:
                raise ValueError(f'self.backend not recognised: {self.backend}')
        else:
            self.last_calculated_metrics = 0.


    def types_of_metrics(self):
        return float


    def tensorboard_tags(self):
        if self.exclude_classes_less_sample_than == 1:
            return 'Video mAP'
        else:
            return 'Video mAP {self.exclude_classes_less_sample_than}+'


    def get_csv_fieldnames(self):
        if self.exclude_classes_less_sample_than == 1:
            return f'{self.split}_vid_mAP'
        else:
            return f'{self.split}_vid_mAP{self.exclude_classes_less_sample_than}+'


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

        if self.exclude_classes_less_sample_than == 1:
            suffix = ''
        else:
            suffix = f'{self.exclude_classes_less_sample_than}+'

        return f'{prefix}vid_mAP{suffix}: {self.last_calculated_metrics:.4f}'


    def plot_legend_labels(self):
        """
        Return:
            either tuple or a single str
        """
        if self.exclude_classes_less_sample_than == 1:
            suffix = ''
        else:
            suffix = f' ({self.exclude_classes_less_sample_than}+ samples)'

        if self.split == 'train':
            return f'Training video mAP{suffix}'
        elif self.split == 'val':
            return f'Validation video mAP{suffix}'
        elif self.split == 'multicropval':
            return f'Multicrop validation video mAP{suffix}'
        else:
            return f'{self.split} video mAP{suffix}'


    def plot_file_basenames(self):
        """
        Return:
            either tuple or a single str
        """
        # output plot file names will be e.g.) accuracy.png/pdf, accuracy_top5.png/pdf, ...
        if self.exclude_classes_less_sample_than == 1:
            suffix = ''
        else:
            suffix = f'_{self.exclude_classes_less_sample_than}+'

        return f'mAP{suffix}'

    @staticmethod
    def is_better(value_1, value_2):
        """Metric comparison function

        params:
            two values of the metric

        return:
            True if value_1 is better. False if value_2 is better or they're equal.
        """
        return value_1 > value_2
