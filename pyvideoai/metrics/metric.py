from abc import ABC, abstractmethod
from typing import Dict
import torch
import numpy as np

import logging
logger = logging.getLogger(__name__)

def is_numpy(instance):
    return type(instance).__module__ == np.__name__


class Metric(ABC):
    """
    Formal structure that all metrics need to follow.
    See example Accuracy / mAP metric classes to see how.
    """
    def clean_data(self):
        self.data = {}              # Store predictions here for later metric calculation
        self.num_classes = None     # prediction length
        self.label_size = None      # either num_classes (multi-label) or 1 (one-label)
        self.last_calculated_metrics = 0.

    def __init__(self, activation=None, video_id_to_label: Dict[int, np.array] = None, video_id_to_label_missing_action: str = 'error', split: str = None):
        """
        video_id_to_label: If given, completely ignore the original label and find labels from this dictionary. Useful when sometimes using different labels to evaluate.
        video_id_to_label_missing_action: If video_id_to_label is given, choose what to do when you can't find the label, in 'error', 'skip', and 'original_label'.
                                        'skip' will ignore the samples. Make sure your metrics support inputs with zero-length tensors.
        """
        self.activation = activation
        self.video_id_to_label = video_id_to_label
        assert video_id_to_label_missing_action in ['error', 'skip', 'noupdate'], f'You chose wrong mode: {video_id_to_label_missing_action}'
        self.video_id_to_label_missing_action = video_id_to_label_missing_action
        self.split = split           # Note that these are permanent after initialisation and shouldn't change with self.clean_data()
        self.clean_data()

    def _apply_activation(self, predictions):
        if self.activation == 'softmax':
            return torch.nn.Softmax(dim=1)(predictions)
        elif self.activation == 'sigmoid':
            return torch.nn.Sigmoid()(predictions)
        elif self.activation is None:
            return predictions
        elif callable(self.activation):
            return self.activation(predictions)
        else:
            raise ValueError(f"activation wrong value: {self.activation}")

    def _check_data_shape(self, video_ids, clip_predictions, labels):
        assert video_ids.shape[0] == clip_predictions.shape[0] == labels.shape[0], "Batch size not equal. video_ids.shape = {}, clip_predictions.shape = {}, labels.shape = {}".format(video_ids.shape, clip_predictions.shape, labels.shape)

        if is_numpy(video_ids):
            video_ids = torch.from_numpy(video_ids)
        if is_numpy(clip_predictions):
            clip_predictions = torch.from_numpy(clip_predictions)
        if is_numpy(labels):
            labels = torch.from_numpy(labels)

        return video_ids, clip_predictions, labels

    def _check_num_classes(self, clip_predictions, labels):
        """Check if the shape of the predictions and labels is consistent.
        """
        # Save the number of classes
        if self.num_classes is None:
            self.num_classes = clip_predictions.shape[1]
        else:
            assert self.num_classes == clip_predictions.shape[1], "Different size of clip predictions from the last times"

        if self.label_size is None:
            self.label_size = 1 if len(labels.shape) == 1 else labels.shape[1]
        else:
            assert self.label_size == (1 if len(labels.shape) == 1 else labels.shape[1]), "Different size of labels from the last times"


    @abstractmethod
    def add_clip_predictions(self, video_ids: torch.FloatTensor,
            clip_predictions: torch.FloatTensor,
            labels: torch.FloatTensor,
            ):
        """By default, we will average the clip predictions with the same video_ids.
        If averaging does not fit your needs, override and redefine this function as you'd like.

        Using this function in a child class:
        ```
        video_ids, clip_predictions, labels = super().add_clip_predictions(video_ids, clip_predictions, labels)
        if video_ids is None:
            return          # sometimes, after filtering out samples of interest, there will be no samples to do anything.
        ```
        """
        video_ids, clip_predictions, labels = self._check_data_shape(video_ids, clip_predictions, labels)

        with torch.no_grad():
            if self.video_id_to_label is not None:
                video_ids_numpy = video_ids.cpu().numpy().astype(int)
                if self.video_id_to_label_missing_action == 'error':
                    labels = torch.tensor(np.array([self.video_id_to_label[vid] for vid in video_ids_numpy]), dtype=video_ids.dtype, device=video_ids.device)
                elif self.video_id_to_label_missing_action == 'original_label':
                    labels = torch.tensor(np.array([self.video_id_to_label[vid] if vid in self.video_id_to_label.keys() else labels[idx] for idx, vid in enumerate(video_ids_numpy)]), dtype=video_ids.dtype, device=video_ids.device)
                else:   # skip
                    labels = torch.tensor(np.array([self.video_id_to_label[vid] for vid in video_ids_numpy if vid in self.video_id_to_label.keys()]), dtype=video_ids.dtype, device=video_ids.device)
                    if len(labels) == 0:
                        return None, None, None
                    video_ids = torch.stack([video_ids[idx] for idx, vid in enumerate(video_ids_numpy) if vid in self.video_id_to_label.keys()])
                    clip_predictions = torch.stack([clip_predictions[idx] for idx, vid in enumerate(video_ids_numpy) if vid in self.video_id_to_label.keys()])


            self._check_num_classes(clip_predictions, labels)
            clip_predictions = self._apply_activation(clip_predictions)

        return video_ids, clip_predictions, labels




    @abstractmethod
    def calculate_metrics(self):
        """
        Return:
            None
            self.last_calculated_metrics: either tuple or a single value (e.g. float)
        """
        self.last_calculated_metrics = 0.


    @abstractmethod
    def types_of_metrics(self):
        """
        Return:
            either tuple or a single type
        """
        return float


    @abstractmethod
    def tensorboard_tags(self):
        """
        Return:
            either tuple or a single str
        """
        return 'Metric'


    @abstractmethod
    def get_csv_fieldnames(self):
        """
        Return:
            either tuple or a single str
        """
        return f'{self.split}_metric'     # like val_acc


    @abstractmethod
    def logging_msg_iter(self):
        """
        Return:
            None to skip logging this metric
            or a single str that combines all self.last_calculated_metrics
        """
        return f'{self.split}_metric: {self.last_calculated_metrics:.5f}'


    @abstractmethod
    def logging_msg_epoch(self):
        """
        Return:
            None to skip logging this metric
            or a single str that combines all self.last_calculated_metrics
        """
        return f'{self.split}_metric: {self.last_calculated_metrics:.5f}'

    @abstractmethod
    def plot_legend_labels(self):
        """
        Return:
            either tuple or a single str
        """
        if self.split == 'train':
            return 'Train metric'
        elif self.split == 'val':
            return 'Validation metric'
        elif self.split == 'multicropval':
            return 'Multicrop validation metric'
        else:
            raise ValueError(f'Unknown split: {self.split}')


    @abstractmethod
    def plot_file_basenames(self):
        """
        Return:
            either tuple or a single str
        """
        return 'metric'     # output plot file names will be metric.png and metric.pdf


    def telegram_report_msg_line(self, exp):
        """
        Params:
            exp (ExperimentBuilder): READONLY. Includes all summary of the train/val stats.
        Return:
            None to skip logging this metric
            or a single str
        """
        if self.split == 'train':
            return None         # Don't print train metric on Telegram
        fieldnames = self.get_csv_fieldnames()
        if isinstance(fieldnames, str):
            fieldnames = (fieldnames,)

        messages = []
        for fieldname in fieldnames:
            if exp.summary[fieldname].count() > 0:
                best_stat = exp.get_best_model_stat(fieldname, self.is_better)
                last_stat = exp.get_last_model_stat(fieldname)
                messages.append(f'Highest (@ epoch {best_stat["epoch"]}) / Last (@ {last_stat["epoch"]}) {fieldname}: {best_stat[fieldname]:.4f} / {last_stat[fieldname]:.4f}')


        return '\n'.join(messages) if len(messages) > 0 else None


    @staticmethod
    def is_better(value_1, value_2):
        """Metric comparison function

        params:
            two values of the metric

        return:
            True if value_1 is better. False if value_2 is better or they're equal.
        """
        return value_1 > value_2


    @abstractmethod
    def __len__(self):
        pass


class ClipMetric(Metric):
    """Clip metric that don't average clip predictions from the same video ID.
    Instead, store all clip predictions independently.
    Example usage: mAP for multilabel classification.
    Note that calculating accuracy don't require saving predictions to the memory, so don't use this for that.
    """
    def add_clip_predictions(self, video_ids, clip_predictions, labels):
        """We will ignore the video_ids and store all the clip predictions independently without aggregating (like averaging).
        """
        video_ids, clip_predictions, labels = super().add_clip_predictions(video_ids, clip_predictions, labels)
        if video_ids is None:
            return

        with torch.no_grad():
            if 'video_ids' in self.data.keys():
                self.data['video_ids'] = torch.cat((self.data['video_ids'], video_ids), dim=0)
                self.data['clip_predictions'] = torch.cat((self.data['clip_predictions'], clip_predictions), dim=0)
                self.data['labels'] = torch.cat((self.data['labels'], labels), dim=0)
            else:
                self.data['video_ids'] = video_ids
                self.data['clip_predictions'] = clip_predictions
                self.data['labels'] = labels

    def get_predictions_torch(self):
        """Return the clip-based predictions in Torch tensor
        """
        return self.data['clip_predictions'].cpu(), self.data['labels'].cpu(), self.data['video_ids'].cpu()

    def get_predictions_numpy(self):
        """Return the video-based predictions in numpy array
        """
        return map(np.array, self.get_predictions_torch())

    def __len__(self):
        return len(self.data['clip_predictions'])


class AverageMetric(Metric):
    """Video metric that averages all clip predictions from the same video ID.
    """
    def add_clip_predictions(self, video_ids, clip_predictions, labels):
        """We will average the clip predictions with the same video_ids.
        """
        video_ids, clip_predictions, labels = super().add_clip_predictions(video_ids, clip_predictions, labels)
        if video_ids is None:
            return

        with torch.no_grad():
            for video_id, clip_prediction, label in zip(video_ids, clip_predictions, labels):
                video_id = round(video_id.item())
                #label = label.item()
                if video_id in self.data.keys():
                    assert torch.equal(self.data[video_id]['label'], label), "Label not consistent between clips from the same video. Video id: %d, label: %s and %s" % (video_id, str(self.data[video_id]['label']), str(label))

                    self.data[video_id]['clip_prediction_accum'] += clip_prediction
                    self.data[video_id]['num_clips'] += 1
                else:
                    self.data[video_id] = {'clip_prediction_accum': clip_prediction, 'label': label, 'num_clips': 1}

    def get_predictions_torch(self):
        """Return the video-based predictions in Torch tensor
        """
        with torch.no_grad():
            num_videos = len(self.data.keys())
            video_predictions = torch.zeros(num_videos, self.num_classes)
            if self.label_size == 1:
                video_labels = torch.zeros(num_videos, dtype=torch.long)
            else:
                video_labels = torch.zeros((num_videos, self.label_size), dtype=torch.long)
            video_ids = torch.zeros(num_videos, dtype=torch.long)
            for b, video_id in enumerate(self.data.keys()):
                video_predictions[b] = self.data[video_id]['clip_prediction_accum'] / self.data[video_id]['num_clips']
                video_labels[b] = self.data[video_id]['label']
                video_ids[b] = video_id

        return video_predictions, video_labels, video_ids

    def get_predictions_numpy(self):
        """Return the video-based predictions in numpy array
        """
        return map(np.array, self.get_predictions_torch())

    def __len__(self):
        num_clips_added = 0
        for key, data in self.data.items():
            num_clips_added += data['num_clips']

        return num_clips_added


class ClipPredictionsGatherer(ClipMetric):
    """Silent, no metric calculation but designed to gather predictions.
    DO NOT USE. Used for val_multiprocess.py --save_predictions only
    """
    def calculate_metrics(self):
        return None

    def types_of_metrics(self):
        return None

    def tensorboard_tags(self):
        return None

    def get_csv_fieldnames(self):
        return None

    def logging_msg_iter(self):
        return None

    def logging_msg_epoch(self):
        return None

    def plot_legend_labels(self):
        return None

    def plot_file_basenames(self):
        return None

    def telegram_report_msg_line(self, exp):
        return None

    def is_better(value_1, value_2):
        return None

class VideoPredictionsGatherer(AverageMetric):
    """Silent, no metric calculation but designed to gather predictions.
    DO NOT USE. Used for val_multiprocess.py --save_predictions only
    """
    def calculate_metrics(self):
        return None

    def types_of_metrics(self):
        return None

    def tensorboard_tags(self):
        return None

    def get_csv_fieldnames(self):
        return None

    def logging_msg_iter(self):
        return None

    def logging_msg_epoch(self):
        return None

    def plot_legend_labels(self):
        return None

    def plot_file_basenames(self):
        return None

    def telegram_report_msg_line(self, exp):
        return None

    def is_better(value_1, value_2):
        return None



class Metrics(dict):
    """A data structure that combines all metrics by splits.
    Example structure:
        self = {'train': [ClipAccuracyMetric()], 'val': [ClipAccuracyMetric()], 'multicropval': [ClipAccuracyMetric(), VideoAccuracyMetric(topk=(1,5))]}
        self.best_metric_split = 'val'
        self.best_metric_index = 0      # indicating that validation clip accuracy is what determines the best model
    """
    def __init__(self, metrics_dict = None, best_metric = None):
        super().__init__()
        self.best_metric_split = None
        self.best_metric_index = None

        if metrics_dict is not None and best_metric is not None:
            self.add_metrics_dict(metrics_dict, best_metric)
        elif (metrics_dict is None) != (best_metric is None):
            raise ValueError('metrics_dict and best_metric must be specified together.')

    def add_metric(self, split, metric: Metric, is_best_metric:bool = False):
        """Adds to the data dictionary, mark the split of the metric, and mark the best metric that determines the best model.
        """
        if is_best_metric:
            if self.best_metric_split is not None or self.best_metric_index is not None:
                raise ValueError('There is already a best metric but is_best_metric is True again. Only one metric can be best metric and this should not happen.')

        if metric.split is None:
            metric.split = split
        else:
#            raise ValueError(f'The split of the metric is already specified ({metric.split}), but trying to add the same metric to another split ({split}), which should never happen.')
            logger.info(f'The split name of the metric {type(metric).__name__} is set to ({metric.split}), although actually it will use "{split}" split to evaluate.')

        if split in self.keys():
            self[split].append(metric)
        else:
            self[split] = [metric]

        if is_best_metric:
            self.best_metric_split = split
            self.best_metric_index = len(self[split]) - 1


    def add_metrics_dict(self, metrics_dict: dict, best_metric: Metric):
        for split, metrics_in_split in metrics_dict.items():
            if metrics_in_split is None or (isinstance(metrics_in_split, list) and len(metrics_in_split) == 0):
                self[split] = []
            elif isinstance(metrics_in_split, Metric):
                # a single metric rather than a list
                is_best_metric = id(metrics_in_split) == id(best_metric)
                self.add_metric(split, metrics_in_split, is_best_metric)
            else:
                # list of metrics
                for metric in metrics_in_split:
                    is_best_metric = id(metric) == id(best_metric)
                    self.add_metric(split, metric, is_best_metric)

        assert self.best_metric_split is not None and self.best_metric_index is not None, 'None of the items in metrics_dict is the best metric that determines the best model. Put best_metric directly in the metrics_dict.'


    def get_best_metric(self):
        return self[self.best_metric_split][self.best_metric_index]

    def get_best_metric_and_fieldname(self):
        best_metric = self.get_best_metric()
        best_metric_fieldname = best_metric.get_csv_fieldnames()
        if isinstance(best_metric_fieldname, tuple):
            if len(best_metric_fieldname) > 1:
                logger.info(f'best_metric returns multiple metric values and PyVideoAI will use the first one: {best_metric_fieldname[0]}.')
            best_metric_fieldname = best_metric_fieldname[0]

        return best_metric, best_metric_fieldname
