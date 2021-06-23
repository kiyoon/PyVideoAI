import torch
import numpy as np

from .AP import mAP
from sklearn.metrics import average_precision_score

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k using PyTorch"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size).item())
    return res

def is_numpy(instance):
    return type(instance).__module__ == np.__name__

class VideoMetrics():
    """Calculate video metrics from multiple clip predictions.
    You pass (video id, clip prediction, label) for all clips and it will average the clip predictions with the same video id and calculate top-k accuracy (one label) / mAP (multi-label).
    """

    def clean_data(self):
        self.data = {}
        self.num_classes = None     # prediction length
        self.label_size = None      # either num_classes (multi-label) or 1 (one-label)

    def __init__(self):
        self.clean_data()

    def add_clip_predictions(self, video_ids, clip_predictions, labels, apply_activation = 'softmax'):
        assert video_ids.shape[0] == clip_predictions.shape[0] == labels.shape[0], "Batch size not equal. video_ids.shape = {}, clip_predictions.shape = {}, labels.shape = {}".format(video_ids.shape, clip_predictions.shape, labels.shape)

        if is_numpy(video_ids):
            video_ids = torch.from_numpy(video_ids)
        if is_numpy(clip_predictions):
            clip_predictions = torch.from_numpy(clip_predictions)
        if is_numpy(labels):
            labels = torch.from_numpy(labels)

        with torch.no_grad():
            # Save the number of classes
            if self.num_classes is None:
                self.num_classes = clip_predictions.shape[1]
            else:
                assert self.num_classes == clip_predictions.shape[1], "Different size of clip predictions from the last times"

            if self.label_size is None:
                self.label_size = 1 if len(labels.shape) == 1 else labels.shape[1]
            else:
                assert self.label_size == (1 if len(labels.shape) == 1 else labels.shape[1]), "Different size of labels from the last times"


            if apply_activation == 'softmax':
                clip_predictions = torch.nn.Softmax(dim=1)(clip_predictions)
            elif apply_activation == 'sigmoid':
                clip_predictions = torch.nn.Sigmoid()(clip_predictions)
            elif apply_activation is None:
                pass
            else:
                raise ValueError(f"apply_activation wrong value: {apply_activation}")

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

    def accuracy(self, topk=(1,)):
        video_predictions, video_labels, _ = self.get_predictions_torch()

        return accuracy(video_predictions, video_labels, topk)

#    def mAP(self, average='macro'):
#        video_predictions, video_labels, video_ids = self.get_predictions_numpy()
#
#        return average_precision_score(video_labels, video_predictions, average=average)

    def mAP(self):
        """ Ignores classes with no positive ground truth
        """
        video_predictions, video_labels, _ = self.get_predictions_numpy()

        return mAP(video_labels, video_predictions)
