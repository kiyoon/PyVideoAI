import torch
from .metric import Metric

EPS = 1e-6


class ClipTop1MultilabelAccuracyMetric(Metric):
    """
    Count as true positive when the top-1 prediction is in one of the multiple ground truth labels.

    Don't need activation softmax for clip accuracy calculation.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def clean_data(self):
        super().clean_data()
        self.num_seen_samples = 0
        self.num_true_positives = 0


    def add_clip_predictions(self, video_ids, clip_predictions, labels):
        video_ids, clip_predictions, labels = super().add_clip_predictions(video_ids, clip_predictions, labels)
        if video_ids is None:
            return      # sometimes, after filtering the samples, there can be no samples to do anything.

        assert labels.dim() == 2, f'labels has to be a 2D tensor with ones and zeros but got {labels.dim()}-D.'

        pred_labels = torch.argmax(clip_predictions, dim=1)
        for pred, label in zip(pred_labels, labels):
            assert label[pred] in [1., 0.], f'Label in Top1 Multilabel Accuracy metric has to be ones and zeros but got {label[pred]}.'

            self.num_seen_samples += 1
            if label[pred] >= 1.-EPS:
                self.num_true_positives += 1


    def calculate_metrics(self):
        self.last_calculated_metrics = self.num_true_positives / self.num_seen_samples if self.num_seen_samples > 0 else 0.

    def types_of_metrics(self):
        return float


    def tensorboard_tags(self):
        return 'Top1 multilabel accuracy'


    def get_csv_fieldnames(self):
        return f'{self.split}_top1_multilabel_accuracy'


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

        message = f'{prefix}top1ML: {self.last_calculated_metrics:.4f}'
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
            return 'Training top1 multilabel accuracy'
        elif self.split == 'val':
            return 'Validation top1 multilabel accuracy'
        elif self.split == 'multicropval':
            return 'Multicrop validation top1 multilabel accuracy'
        else:
            return f'{self.split} top1 multilabel accuracy'


    def plot_file_basenames(self):
        """
        Return:
            either tuple or a single str
        """
        # output plot file names will be e.g.) accuracy.png/pdf, accuracy_top5.png/pdf, ...
        return 'top1_multilabel_accuracy'


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


class ClipTopkMultilabelAccuracyMetric(Metric):
    """
    Count as true positive when the top-k prediction is in one of the k multiple ground truth labels.

    Don't need activation softmax for clip accuracy calculation.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def clean_data(self):
        super().clean_data()
        self.num_seen_samples = 0
        self.num_true_positives = 0


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

            self.num_seen_samples += 1
            _, top_classes = torch.topk(pred, num_labels)
            for top_class in top_classes:
                if top_class in labels_cur_sample:
                    self.num_true_positives += 1
                    break


    def calculate_metrics(self):
        self.last_calculated_metrics = self.num_true_positives / self.num_seen_samples if self.num_seen_samples > 0 else 0.

    def types_of_metrics(self):
        return float


    def tensorboard_tags(self):
        return 'Top-K multilabel accuracy'


    def get_csv_fieldnames(self):
        return f'{self.split}_topk_multilabel_accuracy'


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

        message = f'{prefix}topkML: {self.last_calculated_metrics:.4f}'
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
            return 'Training top-k multilabel accuracy'
        elif self.split == 'val':
            return 'Validation top-k multilabel accuracy'
        elif self.split == 'multicropval':
            return 'Multicrop validation top-k multilabel accuracy'
        else:
            return f'{self.split} top-k multilabel accuracy'


    def plot_file_basenames(self):
        """
        Return:
            either tuple or a single str
        """
        # output plot file names will be e.g.) accuracy.png/pdf, accuracy_top5.png/pdf, ...
        return 'topk_multilabel_accuracy'


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
