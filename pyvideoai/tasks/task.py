from abc import ABC, abstractmethod

import logging
logger = logging.getLogger(__name__)

from ..metrics import Metrics

class Task(ABC):
    def __init__(self):
        pass

    def get_criterion(self, exp_cfg, split):
        return self.get_criterions(exp_cfg, [split])[split]

    def get_criterions(self, exp_cfg, splits):
        if hasattr(exp_cfg, 'criterion'):
            # same criterion for all splits
            raise ValueError('cfg.criterion() is deprecated and it will not work. Please use cfg.get_criterion(split) instead.')
        elif hasattr(exp_cfg, 'get_criterion'):
            return {split: exp_cfg.get_criterion(split) for split in splits}

        default_criterions = {split: self._default_criterion(split) for split in splits}
        logger.info(f'cfg.get_criterions not defined. Using {str(default_criterions)}')
        return default_criterions


    @abstractmethod
    def _default_criterion(self, split):
        """
        Return:
            torch.nn.modules._Loss
        """
        pass


    def get_last_activation(self, exp_cfg):
        if hasattr(exp_cfg, 'last_activation'):
            return exp_cfg.last_activation

        default_last_activation = self._default_last_activation()
        logger.info(f'cfg.last_activation not defined. Using {str(default_last_activation)}')
        return default_last_activation


    @abstractmethod
    def _default_last_activation(self):
        """
        Return:
            callable, or str
        """
        pass
    

    def get_metrics(self, exp_cfg):
        """
        Return:
            metrics (pyvideoai.metrics.Metrics)
        """
        has_best_metric = hasattr(exp_cfg, 'best_metric')
        has_metrics = hasattr(exp_cfg, 'metrics')

        if has_best_metric and has_metrics:
            metrics = Metrics(exp_cfg.metrics, exp_cfg.best_metric)
        elif has_best_metric != has_metrics:
            raise ValueError('exp_cfg has only one of `best_metric` and `metrics`. You must specify either both or none of them.')
        else:   # (not has_best_metric) and (not has_metrics)
            best_metric, metrics_dict = self._default_metrics(self.get_last_activation(exp_cfg))
            metrics = Metrics(metrics_dict, best_metric)

        return metrics


    @abstractmethod
    def _default_metrics(self, activation):
        """
        Return:
            best_metric (pyvideoai.metrics.Metric), metrics_dict (dict of splits as keys, list of Metric as values)
        """
        pass


    def get_predictions_gatherers(self, exp_cfg):
        """
        Predictions_gatherer is a Metric that doesn't log/report anything but accumulates predictions over the dataset.
        
        Return:
            predictions_gatherers (dict of split as keys, predictions_gatherer as values)
        """

        if hasattr(exp_cfg, 'predictions_gatherers'):
            predictions_gatherers = exp_cfg.predictions_gatherers
        else:   # (not has_best_metric) and (not has_metrics)
            predictions_gatherers = self._default_predictions_gatherers(self.get_last_activation(exp_cfg))

        return predictions_gatherers


    @abstractmethod
    def _default_predictions_gatherers(self, activation):
        """
        Return:
            best_metric (pyvideoai.metrics.Metric), metrics_dict (dict of splits as keys, list of Metric as values)
        """
        pass
