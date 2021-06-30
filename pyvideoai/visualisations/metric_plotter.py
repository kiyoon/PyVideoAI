
from experiment_utils.experiment_builder import ExperimentBuilder
from ..metrics import Metric

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
        elif plot_basename in self.basename_to_metrics.keys():
            self.basename_to_metrics[plot_basename].append((plot_legend_label, csv_fieldname))
        else:
            self.basename_to_metrics[plot_basename] = [(plot_legend_label, csv_fieldname)]

    def _add_loss_metrics(self):
        self._add_metric_decomposed('loss', 'Training loss', 'train_loss')
        self._add_metric_decomposed('loss', 'Validation loss', 'val_loss')

    def add_metric(self, metric: Metric) -> None:
        plot_basenames = metric.plot_file_basenames()
        plot_legend_labels = metric.plot_legend_labels()
        csv_fieldnames = metric.get_csv_fieldnames()
        for plot_basename, plot_legend_label, csv_fieldname in zip(plot_basenames, plot_legend_labels, csv_fieldnames):
            self._add_metric_decomposed(plot_basename, plot_legend_label, csv_fieldname)


    def plot(self, exp: ExperimentBuilder):
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
        label = {'train_loss': 'training loss', 'train_acc': 'training accuracy', 'val_loss': 'validation clip loss', 'val_acc': 'validation clip accuracy', 'multi_crop_val_loss': 'multi-crop validation clip loss', 'multi_crop_val_acc': 'multi-crop validation clip acc', 'multi_crop_val_vid_acc_top1': 'multi-crop validation video accuracy top-1', 'multi_crop_val_vid_acc_top5': 'multi-crop validation video accuracy top-5'}

                valid_rows = exp.summary[fieldname].notnull()

                ax_1.plot(exp.summary['epoch'][valid],
                        exp.summary[fieldname][valid], label=plot_legend_label)
                num_plots_in_fig += 1

            if num_plots_in_fig > 0:
                ax_1.legend(loc=0)
                ax_1.set_xlabel('Epoch number')

                fig.tight_layout()

                save_path_wo_ext = os.path.join(exp.plots_dir, plot_basename)
                os.makedirs(os.path.dirname(save_path_wo_ext), exist_ok=True)

                fig.savefig(save_path_wo_exp + '.pdf')
                fig.savefig(save_path_wo_exp + '.png')

                figs.append((plot_basename, fig))

        return figs
