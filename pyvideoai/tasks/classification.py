from .task import Task
from torch import nn

from ..metrics.metric import ClipPredictionsGatherer, VideoPredictionsGatherer
from ..metrics.accuracy import ClipAccuracyMetric, VideoAccuracyMetric
from ..metrics.mAP import Clip_mAPMetric, Video_mAPMetric

import logging
logger = logging.getLogger(__name__)

class SingleLabelClassificationTask(Task):
    def _default_criterion(self, split):
        return nn.CrossEntropyLoss()

    def _default_last_activation(self):
        return 'softmax'

    def _default_metrics(self, activation):
        best_metric = ClipAccuracyMetric(topk=(1,5))
        metrics_dict = {'train': [ClipAccuracyMetric()],
                'val': [best_metric],
                'multicropval': [ClipAccuracyMetric(), VideoAccuracyMetric(topk=(1,5), activation=activation)],
                }
        return best_metric, metrics_dict

    def _default_predictions_gatherers(self, activation):

        return {'val': ClipPredictionsGatherer(activation=activation),
                'multicropval': VideoPredictionsGatherer(activation=activation),
                }


class MultiLabelClassificationTask(Task):
    def _default_criterion(self, split):
        return nn.BCEWithLogitsLoss()

    def _default_last_activation(self):
        return 'sigmoid'

    def _default_metrics(self, activation):
        best_metric = Clip_mAPMetric(activation=activation)
        metrics_dict = {'train': [Clip_mAPMetric(activation=activation)],
                'val': [best_metric],
                'multicropval': [Clip_mAPMetric(activation=activation), Video_mAPMetric(activation=activation)],
                }
        return best_metric, metrics_dict

    def _default_predictions_gatherers(self, activation):

        return {'val': ClipPredictionsGatherer(activation=activation),
                'multicropval': VideoPredictionsGatherer(activation=activation),
                }
