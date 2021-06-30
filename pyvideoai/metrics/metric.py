from abc import *

import torch
import numpy as np
def is_numpy(instance):
    return type(instance).__module__ == np.__name__

class Metric(ABC):
    def clean_data(self):
        self.data = {}              # Store predictions here for later metric calculation
        self.num_classes = None     # prediction length
        self.label_size = None      # either num_classes (multi-label) or 1 (one-label)
        self.last_calculated_metrics = None

    def __init__(self, activation=None):
        self.activation = activation
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
    def add_clip_predictions(self, video_ids, clip_predictions, labels):
        """By default, we will average the clip predictions with the same video_ids.
        If averaging does not fit your needs, override and redefine this function as you'd like.
        """ 
        video_ids, clip_predictions, labels = self._check_data_shape(video_ids, clip_predictions, labels)

        with torch.no_grad():
            self._check_num_classes(clip_predictions, labels)
            clip_predictions = self._apply_activation(clip_predictions)




    @abstractmethod
    def calculate_metrics(self):
        """
        video_predictions, video_labels, _ = self.get_predictions_torch()
        self.last_calculated_metrics = accuracy(video_predictions, video_labels, topk)

        Return:
            either tuple or a single value (e.g. float)
        """
        pass


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
    def get_csv_fieldnames(self, split):
        """
        Return:
            either tuple or a single str 
        """
        return f'{split}_metric'     # like val_acc


    @abstractmethod
    def logging_msg_iter(self, split):
        """
        Return:
            None to skip logging this metric
            or a single str that combines all self.last_calculated_metrics 
        """
        return f'{split}_metric: {self.last_calculated_metrics:.5f}'


    @abstractmethod
    def logging_msg_epoch(self, split):
        """
        Return:
            None to skip logging this metric
            or a single str that combines all self.last_calculated_metrics 
        """
        return f'{split}_metric: {self.last_calculated_metrics:.5f}'
    
    @abstractmethod
    def plot_legend_labels(self, split):
        """
        Return:
            either tuple or a single str 
        """
        if split == 'train':
            return 'Train metric' 
        elif split == 'val':
            return 'Validation metric' 
        elif split == 'multicropval':
            return 'Multicrop validation metric' 
        else:
            raise ValueError(f'Unknown split: {split}')


    @abstractmethod
    def plot_file_basenames(self):
        """
        Return:
            either tuple or a single str 
        """
        return 'metric'     # output plot file names will be metric.png and metric.pdf


    def telegram_report_msg_line(self, split, exp):
        """
        Params:
            exp (ExperimentBuilder): READONLY. Includes all summary of the train/val stats.
        Return:
            None to skip logging this metric
            or a single str
        """
        if split == 'train':
            return None         # Don't print train metric on Telegram
        fieldnames = self.get_csv_fieldnames(split)
        if isinstance(fieldnames, str):
            fieldnames = (fieldnames,)

        messages = []
        for fieldname in fieldnames:
            best_stat = exp.get_best_model_stat(fieldname, self.is_better)
            last_stat = exp.get_last_model_stat(fieldname)
            messages.append(f'Highest (@epoch {best_stat["epoch"]}) / Last (@ {last_stat["epoch"]}) {fieldname}: {best_stat[fieldname]} / {last_stat[fieldname]}')

        return '\n'.join(messages)

    
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
        super().add_clip_predictions(video_ids, clip_predictions, labels)
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
        super().add_clip_predictions(video_ids, clip_predictions, labels)
        with torch.no_grad():
            for video_id, clip_prediction, label in zip(video_ids, clip_predictions, labels):
                video_id = video_id.item()
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
        for data in self.data:
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

    def get_csv_fieldnames(self, split):
        return None

    def logging_msg_iter(self, split):
        return None

    def logging_msg_epoch(self, split):
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

    def get_csv_fieldnames(self, split):
        return None

    def logging_msg_iter(self, split):
        return None

    def logging_msg_epoch(self, split):
        return None
    
    def is_better(value_1, value_2):
        return None
