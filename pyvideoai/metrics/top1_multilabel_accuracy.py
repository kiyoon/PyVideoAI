import torch
import numpy as np
from .metric import Metric



class ClipTop1MultilabelAccuracyMetric(Metric):
    """
    Count as true positive when the top-1 prediction is in one of the multiple ground truth labels.

    Don't need activation softmax for clip accuracy calculation.
    """
    def __init__(self, , activation=None, video_id_to_label: dict[np.array] = None):
        """
        Params:
            class_groups: list of list of class indices. For example, if 0,2 is a head class group and 1,3 is a tail group, [[0,2], [1,3]].
            group_names: name of each group. For example, ['head', 'tail']
        """
        super().__init__(activation=activation, video_id_to_label=video_id_to_label)

    def clean_data(self):
        super().clean_data()
        self.num_seen_samples = 0
        self.num_true_positives = 0


    def add_clip_predictions(self, video_ids, clip_predictions, labels):
        super().add_clip_predictions(video_ids, clip_predictions, labels)

        assert labels.dim() == 2, f'target has to be a 2D tensor with ones and zeros but got {target.dim()}-D.'

        pred_labels = torch.argmax(clip_predictions, dim=1)
        for pred, label in zip(pred_labels, labels):
            assert label[pred] in [1., 0.], f'Label in Top1 Multilabel Accuracy metric has to be ones and zeros but got {label[pred]}'.

            self.num_seen_samples += 1
            if label[pred] == 1.:
                self.num_true_positives += 1


    def calculate_metrics(self):
        self.last_calculated_metrics = self.num_true_positives / self.num_seen_samples 

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

        message = f'{prefix}top1multiacc: {self.last_calculated_metrics:.4f}'
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
            return f'Training top1 multilabel accuracy'
        elif self.split == 'val':
            return f'Validation top1 multilabel accuracy'
        elif self.split == 'multicropval':
            return f'Multicrop validation top1 multilabel accuracy'
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

