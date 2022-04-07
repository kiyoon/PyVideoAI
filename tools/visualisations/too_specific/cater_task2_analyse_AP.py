
import dataset_configs
import model_configs
import argparse
import config
from experiment_utils.argparse_utils import add_exp_arguments
def get_parser():
    parser = argparse.ArgumentParser(description="Analyse using per class AP.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_exp_arguments(parser, dataset_configs.available_datasets, model_configs.available_models, root_default=config.DEFAULT_EXPERIMENT_ROOT, dataset_default='cater_task2', model_default='trn_resnet50', name_default='test')
    parser.add_argument("-l", "--load_epoch", type=int, default=None, help="Load from checkpoint. Set to -1 to load from the last checkpoint, and to -2 to load best model in terms of val_acc.")
    parser.add_argument("-m", "--mode", type=str, default="oneclip", choices=["oneclip", "multicrop"],  help="Mode used for run_val.py")

    return parser

parser = get_parser()
args = parser.parse_args()

from experiment_utils.csv_to_dict import csv_to_dict
from experiment_utils import ExperimentBuilder
import os
import pandas as pd
import matplotlib.pyplot as plt
#matplotlib.use('TKAgg')

from video_datasets_api.cater.class_keys import ACTIONS, OBJECTS, class_keys_task1
OBJECT_AND_ACTIONS = class_keys_task1()

if __name__ == '__main__':
    dataset_cfg = dataset_configs.load_cfg(args.dataset)

    perform_multicropval=True       # when loading, assume there was multicropval. Even if there was not, having more CSV field information doesn't hurt.
    if dataset_cfg.task == 'singlelabel_classification':
        multilabel = False
        if args.mode == 'oneclip':
            best_metric_field = 'val_acc'
        else:   # multicrop
            best_metric_field = 'multi_crop_val_vid_acc_top1'
        summary_fieldnames, summary_fieldtypes = ExperimentBuilder.return_fields_singlelabel(multicropval = perform_multicropval)
    elif dataset_cfg.task == 'multilabel_classification':
        multilabel = True
        if args.mode == 'oneclip':
            best_metric_field = 'val_vid_mAP'
        else:   # multicrop
            best_metric_field = 'multi_crop_val_vid_mAP'
        summary_fieldnames, summary_fieldtypes = ExperimentBuilder.return_fields_multilabel(multicropval = perform_multicropval)
    else:
        raise ValueError(f"Not recognised dataset_cfg.task: {dataset_cfg.task}")

    exp = ExperimentBuilder(args.experiment_root, args.dataset, args.model, args.experiment_name, args.subfolder_name, summary_fieldnames = summary_fieldnames, summary_fieldtypes = summary_fieldtypes)
    exp.load_summary()

    if args.load_epoch == -1:
        load_epoch = int(exp.summary['epoch'][-1])
    elif args.load_epoch == -2:
        load_epoch = int(exp.get_best_model_stat(best_metric_field)['epoch'])
    elif args.load_epoch >= 0:
        load_epoch = args.load_epoch
    else:
        raise ValueError(f"Wrong args.load_epoch value: {args.load_epoch:d}")

    per_class_AP_path = os.path.join(exp.plots_dir, f'per_class_AP-epoch_{load_epoch:04d}_{args.mode}val.csv')
    per_class_AP, _ = csv_to_dict(per_class_AP_path, {'class_key': str, 'AP': float, 'TP': int, 'num_samples_in_val': int, 'num_samples_in_train': int})
    per_class_AP = pd.DataFrame.from_dict(per_class_AP)

    output_dir = os.path.join(exp.plots_dir, f'AP_analysis-epoch_{load_epoch:04d}_{args.mode}val.csv')
    print(f"Saving to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    def plot_grouped_analysis(class_key_contains, title, num_train_samples_threshold=0, xtickrotation=0, figsize=None):
        means = []
        stds = []
        lens = []
        sum_num_train_samples = []
        for contain_str in class_key_contains:
            num_train_cond = per_class_AP['num_samples_in_train'] > num_train_samples_threshold
            positive_AP_cond = per_class_AP['AP'] >= 0
            class_filter_cond = per_class_AP['class_key'].str.contains(contain_str)

            filtered_AP = per_class_AP.loc[num_train_cond & positive_AP_cond & class_filter_cond]
            #print(filtered_AP)

            lens.append(len(filtered_AP))
            means.append(filtered_AP['AP'].mean())
            stds.append(filtered_AP['AP'].std())
            sum_num_train_samples.append(filtered_AP['num_samples_in_train'].sum())

        x = list(range(len(class_key_contains)))

        fig, ax1 = plt.subplots(figsize=figsize)
        color = 'tab:red'
        ax1.set_xticks(x)
        ax1.set_xticklabels(class_key_contains, rotation=xtickrotation)
        ax1.set_ylim([0,1])
        ax1.set_ylabel('AP mean/std', color=color)
        ax1.errorbar(x, means, stds, linestyle='None', marker='^', color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('# filtered classes (x)\n# train samples / 1000 (o)', color=color)  # we already handled the x-label with ax1
        ax2.set_ylim([0,200])
        ax2.scatter(x, lens, color=color, marker='x')
        ax2.scatter(x, [x / 1000 for x in sum_num_train_samples], color=color, marker='o')
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.title(f'AP based on {title}')
        plt.figtext(0.99, 0.01, f'filters: AP>=0 & #samples_train > {num_train_samples_threshold}', horizontalalignment='right')
        plt.savefig(os.path.join(output_dir, f"{title}_{num_train_samples_threshold}.pdf"))
        plt.savefig(os.path.join(output_dir, f"{title}_{num_train_samples_threshold}.png"))

    ORDERING = ['before', 'during']
    plot_grouped_analysis(ORDERING, "ORDERING", 0)
    plot_grouped_analysis(ACTIONS, "ACTIONS", 0)
    plot_grouped_analysis(OBJECTS, "OBJECTS", 0)
    plot_grouped_analysis(OBJECT_AND_ACTIONS, "OBJECT_AND_ACTIONS", 0, 90, (7,7))
    plot_grouped_analysis(ORDERING, "ORDERING", 100)
    plot_grouped_analysis(ACTIONS, "ACTIONS", 100)
    plot_grouped_analysis(OBJECTS, "OBJECTS", 100)
    plot_grouped_analysis(OBJECT_AND_ACTIONS, "OBJECT_AND_ACTIONS", 100, 90, (7,7))
    plot_grouped_analysis(ORDERING, "ORDERING", 300)
    plot_grouped_analysis(ACTIONS, "ACTIONS", 300)
    plot_grouped_analysis(OBJECTS, "OBJECTS", 300)
    plot_grouped_analysis(OBJECT_AND_ACTIONS, "OBJECT_AND_ACTIONS", 300, 90, (7,7))
    plot_grouped_analysis(ORDERING, "ORDERING", 500)
    plot_grouped_analysis(ACTIONS, "ACTIONS", 500)
    plot_grouped_analysis(OBJECTS, "OBJECTS", 500)
    plot_grouped_analysis(OBJECT_AND_ACTIONS, "OBJECT_AND_ACTIONS", 500, 90, (7,7))
