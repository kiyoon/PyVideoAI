
from experiment_utils.experiment_builder import ExperimentBuilder
from ..metrics import Metrics
import torch

from .. import __version__

import logging
logger = logging.getLogger(__name__)

class DefaultTelegramReporter:
    def __init__(self, include_wandb_url: bool = True, include_version: bool = True, include_exp_rootdir: bool = False, ignore_figures: bool = False):
        self.include_wandb_url = include_wandb_url
        self.include_version = include_version
        self.include_exp_rootdir = include_exp_rootdir
        self.ignore_figures = ignore_figures

    def report(self, metrics: Metrics, exp: ExperimentBuilder, figs) -> None:
        telegram_report_msgs = []
        if self.include_wandb_url:
            try:
                import wandb
                if wandb.run is not None:
                    telegram_report_msgs.append(f'W&B Project: {wandb.run.get_project_url()}')
                    telegram_report_msgs.append(f'W&B Run: {wandb.run.get_url()}')
                else:
                    logger.debug('`wandb.run` does not exist. Forgot to call `wandb.init()`? Skipping reporting W&B URLs to Telegram.')
            except ImportError:
                logger.debug('Package `wandb` not installed. Skipping reporting W&B URLs to Telegram.')
        if self.include_version:
            telegram_report_msgs.append(f'PyTorch={torch.__version__}, PyVideoAI={__version__}')
        if self.include_exp_rootdir:
            telegram_report_msgs.append(f'Experiment root: {exp.experiment_root}')

        for split, metrics_in_split in metrics.items():
            for metric in metrics_in_split:
                msg_line = metric.telegram_report_msg_line(exp)
                if msg_line is not None:
                    telegram_report_msgs.append(msg_line)

        telegram_report_msgs.append(exp.time_summary_to_text())

        final_message = '\n'.join(telegram_report_msgs)
        exp.tg_send_text_with_expname(final_message)

        if not self.ignore_figures:
            for plot_basename, fig in figs:
                exp.tg_send_matplotlib_fig(fig)
