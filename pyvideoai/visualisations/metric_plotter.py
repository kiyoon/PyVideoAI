
from experiment_utils.experiment_builder import ExperimentBuilder
from ..metrics import Metric, Metrics

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')       # Default TKinter backend instantiates windows even when saving the plot to files, causing problems.


import pandas as pd
import os

class DefaultMetricPlotter:
    def __init__(self):
        # data structure e.g.: {'accuracy': [(plot_legend_label, csv_fieldname)], 'loss': [(...)], 'accuracy_top5': [(...), (...), (...)]}
        self.basename_to_metrics = {}
        self._add_loss_metrics()

    def _add_metric_decomposed(self, plot_basename: str, plot_legend_label: str, csv_fieldname: str) -> None:
        if plot_basename in self.basename_to_metrics.keys():
            self.basename_to_metrics[plot_basename].append((plot_legend_label, csv_fieldname))
        else:
            self.basename_to_metrics[plot_basename] = [(plot_legend_label, csv_fieldname)]

    def _add_loss_metrics(self):
        self._add_metric_decomposed('loss', 'Training loss', 'train_loss')
        self._add_metric_decomposed('loss', 'Validation loss', 'val_loss')

    def add_metric(self, metric: Metric) -> None:
        plot_basenames = metric.plot_file_basenames()
        if not isinstance(plot_basenames, tuple):
            plot_basenames = (plot_basenames,)
        plot_legend_labels = metric.plot_legend_labels()
        if not isinstance(plot_legend_labels, tuple):
            plot_legend_labels = (plot_legend_labels,)
        csv_fieldnames = metric.get_csv_fieldnames()
        if not isinstance(csv_fieldnames, tuple):
            csv_fieldnames = (csv_fieldnames,)
        for plot_basename, plot_legend_label, csv_fieldname in zip(plot_basenames, plot_legend_labels, csv_fieldnames):
            self._add_metric_decomposed(plot_basename, plot_legend_label, csv_fieldname)

    def add_metrics(self, metrics: Metrics) -> None:
        for split, metrics_in_split in metrics.items():
            for metric in metrics_in_split:
                self.add_metric(metric)

    def plot(self, exp: ExperimentBuilder) -> list:
        plt.rcParams.update({'font.size': 16})
        figs = []       # list of (plot_basename, fig)
        for plot_basename, metric_infos in self.basename_to_metrics.items():
            num_plots_in_fig = 0
            fig = plt.figure(figsize=(8, 4)) 

            ax_1 = fig.add_subplot(111)
        #    ax_1.set_xlim([0,100])
        #    ax_1.set_ylim([0,1])
            ax_1.set_xlim(auto=True)
            ax_1.set_ylim(auto=True)

            for plot_legend_label, fieldname in metric_infos:
                if exp.summary[fieldname].count() > 0:      # count non-NaN values
                    valid_rows = exp.summary[fieldname].notnull()

                    ax_1.plot(exp.summary['epoch'][valid_rows],
                            exp.summary[fieldname][valid_rows], label=plot_legend_label)
                    num_plots_in_fig += 1

            if num_plots_in_fig > 0:
                ax_1.legend(loc=0)
                ax_1.set_xlabel('Epoch number')

                fig.tight_layout()

                save_path_wo_ext = os.path.join(exp.plots_dir, plot_basename)
                os.makedirs(os.path.dirname(save_path_wo_ext), exist_ok=True)

                fig.savefig(save_path_wo_ext + '.pdf')
                fig.savefig(save_path_wo_ext + '.png')

                figs.append((plot_basename, fig))
            else:
                # Eventually, no plot is generated. Close unused figure
                plt.close(fig)

        return figs
