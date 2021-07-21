
from experiment_utils.experiment_builder import ExperimentBuilder
from ..metrics import Metrics
import socket

from .. import __version__

class DefaultTelegramReporter:
    def report(self, metrics: Metrics, exp: ExperimentBuilder, figs) -> None:
        #telegram_report_msgs = [f'Running on {socket.gethostname()}, Plots at epoch {exp.summary["epoch"].iloc[-1]:d}']
        telegram_report_msgs = [f'Running on {socket.gethostname()}, PyVideoAI=={__version__}']
        
        for split, metrics_in_split in metrics.items():
            for metric in metrics_in_split:
                msg_line = metric.telegram_report_msg_line(exp) 
                if msg_line is not None:
                    telegram_report_msgs.append(msg_line)

        telegram_report_msgs.append(exp.time_summary_to_text())

        final_message = '\n'.join(telegram_report_msgs) 
        exp.tg_send_text_with_expname(final_message)

        for plot_basename, fig in figs:
            exp.tg_send_matplotlib_fig(fig)
