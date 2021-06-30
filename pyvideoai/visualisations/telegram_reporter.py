
from experiment_utils.experiment_builder import ExperimentBuilder
from ..metrics import Metrics

class DefaultTelegramReporter:
    def report(self, metrics: Metrics, exp: ExperimentBuilder, figs) -> None:
        telegram_report_msgs = [f'Plots at epoch {exp.summary["epoch"]:d}']
        
        for split, metrics_in_split in metrics.items():
            for metric in metrics_in_split:
                telegram_report_msgs.append(metric.telegram_report_msg_line(exp))

        telegram_report_msgs.append(exp.time_summary_to_text())

        final_message = '\n'.join(telegram_report_msgs) 
        exp.tg_send_text_with_expname(final_message)

        for plot_basename, fig in figs:
            exp.tg_send_matplotlib_fig(fig)
