from .task import Task
from torch import nn

from ..metrics import Metrics
from ..metrics.accuracy import ClipAccuracyMetric, VideoAccuracyMetric

import logging
logger = logging.getLogger(__name__)

class SingleLabelClassificationTask(Task):
    def _default_criterion(self):
        return nn.CrossEntropyLoss()

    def _default_metrics(self):
        best_metric = ClipAccuracyMetric()
        metrics_dict = {'train': [ClipAccuracyMetric()],
                'val': [best_metric],
                'multicropval': [ClipAccuracyMetric(), VideoAccuracyMetric(topk=(1,5))],
                }
        return best_metric, metrics_dict



class MultiLabelClassificationTask(Task):
    def _default_criterion(self):
        return nn.BCEWithLogitsLoss()
