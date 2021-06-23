from abc import *

import torch
import numpy as np
def is_numpy(instance):
    return type(instance).__module__ == np.__name__

class Metric(metaclass=ABCMeta):
    def clean_data(self):
        self.data = {}              # Store predictions here for later metric calculation
        self.num_classes = None     # prediction length
        self.label_size = None      # either num_classes (multi-label) or 1 (one-label)

    def __init__(self, prefix=None):
        """
        args:
            prefix (str): prefix of the metric name (usually, "val_" or "multicropval_" to make "acc" -> "val_acc" for the CSV fieldnames)
        """
        self.clean_data()
        self.prefix = prefix

    def _apply_activation(self, predictions, activation):
        if activation == 'softmax':
            return torch.nn.Softmax(dim=1)(predictions)
        elif activation == 'sigmoid':
            return torch.nn.Sigmoid()(predictions)
        elif activation is None:
            return clip_predictions
        elif callable(activation):
            return activation(predictions)
        else:
            raise ValueError(f"activation wrong value: {activation}")

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


    def add_clip_predictions(self, video_ids, clip_predictions, labels, apply_activation = 'softmax'):
        """By default, we will average the clip predictions with the same video_ids.
        If averaging does not fit your needs, redefine this function as you'd like.
        """ 
        video_ids, clip_predictions, labels = self._check_data_shape(video_ids, clip_predictions, labels)

        with torch.no_grad():
            self._check_num_classes(clip_predictions, labels)
            clip_predictions = self._apply_activation(clip_predictions, apply_activation)

            for video_id, clip_prediction, label in zip(video_ids, clip_predictions, labels):
                video_id = video_id.item()
                #label = label.item()
                if video_id in self.data.keys():
                    assert torch.equal(self.data[video_id]['label'], label), "Label not consistent between clips from the same video. Video id: %d, label: %s and %s" % (video_id, str(self.data[video_id]['label']), str(label))

                    self.data[video_id]['clip_prediction_accum'] += clip_prediction
                    self.data[video_id]['num_clips'] += 1
                else:
                    self.data[video_id] = {'clip_prediction_accum': clip_prediction, 'label': label, 'num_clips': 1}

    @abstractmethod
    def get_predictions_torch(self):
        """Return the video-based predictions in Torch tensor
        """
        #return video_predictions, video_labels, video_ids
        return None, None, None


    @abstractmethod
    def calculate_metrics(self, topk=(1,)):
        video_predictions, video_labels, _ = self.get_predictions_torch()
        return accuracy(video_predictions, video_labels, topk)


    @abstractmethod
    def get_types_of_metrics(self):
        return float


    @abstractmethod
    def get_tensorboard_tags(self):
        # DO NOT add self.prefix
        return 'Metric'


    @abstractmethod
    def get_csv_fieldnames(self):
        return f'{self.prefix}metric'     # like val_acc


    @abstractmethod
    def get_logging_format(self):
        return f'{self.prefix}metric: {{:.5f}}'


    def get_predictions_numpy(self):
        """Return the video-based predictions in numpy array 
        """
        return map(np.array, self.get_predictions_torch())

    def __len__(self):
        num_clips_added = 0
        for data in self.data:
            num_clips_added += data['num_clips']

        return num_clips_added
