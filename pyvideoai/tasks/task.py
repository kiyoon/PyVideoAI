from abc import *

import logging
logger = logging.getLogger(__name__)

from ..metrics import Metrics

class Task(ABC):
    def __init__(self):
        pass

    def get_criterion(self, exp_cfg):
        if hasattr(exp_cfg, 'criterion'):
            return exp_cfg.criterion()

        default_criterion = self._default_criterion()
        logger.info(f'cfg.criterion not defined. Using {str(default_criterion)}')
        return default_criterion


    @abstractmethod
    def _default_criterion(self):
        """
        Return:
            torch.nn.modules._Loss
        """
        pass


    def get_metrics(self, exp_cfg):
        has_best_metric = hasattr(exp_cfg, 'best_metric')
        has_metrics = hasattr(exp_cfg, 'metrics')

        if has_best_metric and has_metrics:
            metrics = Metrics(exp_cfg.metrics, exp_cfg.best_metric)
        elif has_best_metric != has_metrics:
            raise ValueError('exp_cfg has only one of `best_metric` and `metrics`. You must specify either both or none of them.')
        else:   # (not has_best_metric) and (not has_metrics)
            best_metric, metrics_dict = self._default_metrics()
            metrics = Metrics(metrics_dict, best_metric)

        return metrics


    @abstractmethod
    def _default_metrics(self):
        """
        Return:
            best_metric (pyvideoai.metrics.Metric), metrics_dict (dict of splits as keys, list of Metric as values)
        """
        pass
