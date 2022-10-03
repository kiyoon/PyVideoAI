import torch
from .metric import Metric



class ClipTopSetMultilabelAccuracyMetric(Metric):
    """
    Multi-label accuracy as defined in Learning Visual Actions Using Multiple Verb-Only Labels (Wray et al. 2019).
    For each sample, if there are k labels, get top-k predictions and see how many of them are in the ground truth.
    Then average all the accuracies per clip.

    Don't need activation softmax for clip accuracy calculation.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def clean_data(self):
        super().clean_data()
        self.accuracy_per_clip = []


    def add_clip_predictions(self, video_ids, clip_predictions, labels: torch.Tensor):
        video_ids, clip_predictions, labels = super().add_clip_predictions(video_ids, clip_predictions, labels)
        if video_ids is None:
            return      # sometimes, after filtering the samples, there can be no samples to do anything.

        assert labels.dim() == 2, f'labels has to be a 2D tensor with ones and zeros but got {labels.dim()}-D.'


        for pred, label in zip(clip_predictions, labels):
            label_indices = (label == 1.).nonzero(as_tuple=False)
            num_labels = len(label_indices)
            _, pred_top_indices = torch.topk(pred, num_labels, largest=True)
            num_true_preds = 0
            for pred_index in pred_top_indices:
                if pred_index in label_indices:
                    num_true_preds += 1

            accuracy_this_clip = num_true_preds / num_labels
            self.accuracy_per_clip.append(accuracy_this_clip)


    def calculate_metrics(self):
        self.last_calculated_metrics = sum(self.accuracy_per_clip) / len(self) if len(self) > 0 else 0.

    def types_of_metrics(self):
        return float


    def tensorboard_tags(self):
        return 'Top-set multilabel accuracy'


    def get_csv_fieldnames(self):
        return f'{self.split}_topset_multilabel_accuracy'


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

        message = f'{prefix}topsetML: {self.last_calculated_metrics:.4f}'
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
            return 'Training Top-set multilabel accuracy'
        elif self.split == 'val':
            return 'Validation Top-set multilabel accuracy'
        elif self.split == 'multicropval':
            return 'Multicrop validation Top-set multilabel accuracy'
        else:
            return f'{self.split} Top-set multilabel accuracy'


    def plot_file_basenames(self):
        """
        Return:
            either tuple or a single str
        """
        # output plot file names will be e.g.) accuracy.png/pdf, accuracy_top5.png/pdf, ...
        return 'topset_ml_accuracy'


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
