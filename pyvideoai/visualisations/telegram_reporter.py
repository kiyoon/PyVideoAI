
from experiment_utils.experiment_builder import ExperimentBuilder
from ..metrics import Metrics
import socket
import torch

from .. import __version__

class DefaultTelegramReporter:
    def __init__(self, include_version: bool = True, include_exp_rootdir: bool = False, ignore_figures: bool = False):
        self.include_version = include_version
        self.include_exp_rootdir = include_exp_rootdir
        self.ignore_figures = ignore_figures

    def report(self, metrics: Metrics, exp: ExperimentBuilder, figs) -> None:
        #telegram_report_msgs = [f'Running on {socket.gethostname()}, Plots at epoch {exp.summary["epoch"].iloc[-1]:d}']
        telegram_report_msgs = []
        if self.include_version:
            telegram_report_msgs.append(f'PyTorch=={torch.__version__}, PyVideoAI=={__version__}')
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
