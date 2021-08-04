"""
Build confusion matrix
"""

import argparse
import pickle
import csv
import logging
import numpy as np
import os
import sys


FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

import torch
from torch import nn
from experiment_utils.csv_to_dict import csv_to_dict
from experiment_utils.argparse_utils import add_exp_arguments
from experiment_utils import ExperimentBuilder
import dataset_configs
import model_configs
import exp_configs
import time

from torch.utils.tensorboard import SummaryWriter

from pyvideoai.config import DEFAULT_EXPERIMENT_ROOT
from pyvideoai.tasks import SingleLabelClassificationTask 


def generate_remapped_labels(num_classes, labels_to_keep):
    new_label = 0

    remapped_labels = []

    num_classes_after_removal = len(labels_to_keep)
    assert num_classes_after_removal > 0

    for old_label in range(num_classes):
        if old_label not in labels_to_keep:
            remapped_labels.append(num_classes_after_removal)   # append `other`
        else:
            remapped_labels.append(new_label)
            new_label += 1

    return np.array(remapped_labels)


def remap_class_keys(class_keys, remapped_labels, num_classes_after_removal):
    new_class_keys = [''] * (num_classes_after_removal+1)#{{{
    new_class_keys[-1] = 'other'
    for i, new_label in enumerate(remapped_labels):
        new_class_keys[new_label] = class_keys[i]

    new_class_keys[-1] = 'other'

    return pd.DataFrame(new_class_keys, columns = ['class_key'])['class_key']#}}}



def remap_labels(labels, remapped_labels):
    def convert_func(label):
        return remapped_labels[label]

    return np.vectorize(convert_func)(labels)


def shrink_labels(num_classes, labels_to_keep, class_keys, pred_labels, video_labels, sort_labels, class_frequency_in_train):
    num_classes_after_removal = len(labels_to_keep)

    remapped_labels = generate_remapped_labels(num_classes, labels_to_keep)
    new_class_keys = remap_class_keys(class_keys, remapped_labels, num_classes_after_removal)
    new_pred_labels = remap_labels(pred_labels, remapped_labels)
    new_video_labels = remap_labels(video_labels, remapped_labels)
    new_sort_labels = remap_labels(sort_labels, remapped_labels)
    # remove "other" on sort_labels and put it at the end
    new_sort_labels = new_sort_labels[new_sort_labels < num_classes_after_removal]
    new_sort_labels = np.append(new_sort_labels, num_classes_after_removal)

    new_class_frequency_in_train = np.zeros(num_classes_after_removal+1, dtype=np.int32)
    for i, num in enumerate(class_frequency_in_train):
        new_class_frequency_in_train[remapped_labels[i]] += num

    return new_class_keys, new_pred_labels, new_video_labels, new_sort_labels, new_class_frequency_in_train



def generate_confusion_matrix(args, sort_method, dataset_cfg, output_dir, video_predictions, video_labels, tb_writer, shrink=False):
    """save confusion matrix"""


    '''
    for i in range(10):
        print(video_predictions[i].max())
        print(video_predictions[i].min())
        print(video_labels[i])
        print()
    '''

    class_indices = range(dataset_cfg.num_classes)
    class_keys = dataset_cfg.class_keys 
    
    pred_labels = np.argmax(video_predictions, axis=1)
    cm = confusion_matrix(video_labels, pred_labels, labels = class_indices)
    class_frequency_in_train = dataset_cfg.count_train_class_frequency()

    num_samples_per_target = cm.sum(axis=1)
    if sort_method == 'val_class_frequency':
        # sort by class frequency in validation data (video-based)
        sort_labels = num_samples_per_target.argsort()[::-1]

    elif sort_method == 'train_class_frequency':
        # sort by class frequency in training data (video-based)
        sort_labels = class_frequency_in_train.argsort()[::-1]

    elif sort_method == 'val_per_class_accuracy':
        # sort by accuracy per class (video-based)
        cm = normalize(cm, axis=1, norm='l1')     # row (true labels) will sum to 1.
        sort_labels = cm.diagonal().argsort()[::-1]

    else:
        raise ValueError('Wrong sort_method')


    if shrink:
        # remove the ones with too few val samples
        # whilst keeping the top 20 classes
        labels_to_keep = []
        
        for i in sort_labels:
            if num_samples_per_target[i] >= 20:
                labels_to_keep.append(i)
            if len(labels_to_keep) >= 20:
                break

        class_keys, pred_labels, video_labels, sort_labels, class_frequency_in_train = shrink_labels(dataset_cfg.num_classes, labels_to_keep, class_keys, pred_labels, video_labels, sort_labels, class_frequency_in_train)
        ##

    cm_sorted = confusion_matrix(video_labels, pred_labels, labels = sort_labels)
    num_samples_per_target = cm_sorted.sum(axis=1)  # class frequency in val
    num_correct_pred_per_target = cm_sorted.diagonal()
    class_frequency_in_train = class_frequency_in_train[sort_labels]

    if args.normalise:
        cm_sorted = normalize(cm_sorted, axis=1, norm='l1')     # row (true labels) will sum to 1.

    # Generate visualisation and summary files

    out_dir = os.path.join(output_dir, 'sort_%s' % sort_method)
    os.makedirs(out_dir, exist_ok = True)

    df_cm = pd.DataFrame(cm_sorted, class_keys[sort_labels],
                              class_keys[sort_labels])

    if False:
        # old settings

        fig = plt.figure(figsize = (350,250))
        ax = fig.add_subplot(111)
        # x label on top
        ax.xaxis.tick_top()

        sn.set(font_scale=10)#for label size
        sn_plot = sn.heatmap(df_cm, annot=False, annot_kws={"size": 12}, cmap="YlGnBu", square=True, vmin=0, vmax=1)# font size
        plt.xlabel('Predicted', fontsize=300)
        plt.ylabel('Target', fontsize=300)

        # This sets the yticks "upright" with 0, as opposed to sideways with 90.
        plt.yticks(fontsize=50, rotation=0) 
        plt.xticks(fontsize=50, rotation=90) 

        # here set the colorbar labelsize by 20
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=20)

    fig = dataset_cfg.plot_confusion_matrix(df_cm, shrink=shrink)
    shrink_str = 'shrinked_' if shrink else ''

    logger.info(f'Saving confusion matrix to {out_dir}')
    plt.savefig('%s/%sconfusion.pdf' % (out_dir, shrink_str))
    plt.savefig('%s/%sconfusion.png' % (out_dir, shrink_str))

    tag = f'{shrink_str}sort_{sort_method}'
    logger.info(f'Saving confusion matrix to TensorBoard tagged {tag}')
    tb_writer.add_figure(tag, fig)

    with open('%s/%sper_class_accuracy.csv' % (out_dir, shrink_str), mode='w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=str(','), quotechar=str('"'), quoting=csv.QUOTE_MINIMAL)

        csvwriter.writerow(['class_key', 'accuracy (%)', 'num_correct_pred', 'num_samples_in_val', 'num_samples_in_train'])

        for class_label, num_correct_pred, num_samples_in_val, num_samples_in_train in zip(class_keys[sort_labels], num_correct_pred_per_target, num_samples_per_target, class_frequency_in_train):
            csvwriter.writerow([class_label, float(num_correct_pred) / num_samples_in_val * 100 if num_samples_in_val != 0 else 'NaN', num_correct_pred, num_samples_in_val, num_samples_in_train])


def main():
    parser = argparse.ArgumentParser(
        description='Load predictions, and generating per-class accuracy and confusion matrix')

    add_exp_arguments(parser, dataset_configs.available_datasets, model_configs.available_models, root_default=DEFAULT_EXPERIMENT_ROOT, dataset_default='hmdb', model_default='i3d_resnet50', name_default='crop224_8x8_1scrop5tcrop_hmdbsplit1pretrained',
            dataset_channel_choices=dataset_configs.available_channels, model_channel_choices=model_configs.available_channels, exp_channel_choices=exp_configs.available_channels)
    parser.add_argument(
        '--no_normalise', action='store_false', dest='normalise', help='Do not normalise the confusion matrix')
    parser.add_argument("-l", "--load_epoch", type=int, default=None, help="Load from checkpoint. Set to -1 to load from the last checkpoint, and to -2 to load best model in terms of val_acc.")
    parser.add_argument("-m", "--mode", type=str, default="oneclip", choices=["oneclip", "multicrop"],  help="Mode used for run_val.py")
    parser.add_argument(
        '--sort_method', type=str, default='all', choices=['train_class_frequency', 'val_class_frequency', 'val_per_class_accuracy', 'all'], help='Sorting method')
    parser.add_argument("-v", "--version", type=str, default='auto', help="Experiment version (`auto` or integer). `auto` chooses the last version.")

    args = parser.parse_args()

    cfg = exp_configs.load_cfg(args.dataset, args.model, args.experiment_name, args.dataset_channel, args.model_channel, args.experiment_channel)
    metrics = cfg.dataset_cfg.task.get_metrics(cfg)

    if not isinstance(cfg.dataset_cfg.task, SingleLabelClassificationTask):
        logger.error(f'Only supports single label classification but got {dataset_cfg.task} task. Exiting..')
        return

    if args.version == 'auto':
        _expversion = -2    # last version (do not create new)
    else:
        _expversion = int(args.version)

    summary_fieldnames, summary_fieldtypes = ExperimentBuilder.return_fields_from_metrics(metrics)
    exp = ExperimentBuilder(args.experiment_root, args.dataset, args.model, args.experiment_name, summary_fieldnames = summary_fieldnames, summary_fieldtypes = summary_fieldtypes, version = _expversion)

    if args.load_epoch == -1:
        exp.load_summary()
        load_epoch = int(exp.summary['epoch'][-1])
    elif args.load_epoch == -2:
        exp.load_summary()
        best_metric = metrics.get_best_metric()
        best_metric_fieldname = best_metric.get_csv_fieldnames()
        best_metric_is_better = best_metric.is_better
        if isinstance(best_metric_fieldname, tuple):
            if len(best_metric_fieldname) > 1:
                logger.warn(f'best_metric returns multiple metric values and PyVideoAI will use the first one: {best_metric_fieldname[0]}.')
            best_metric_fieldname = best_metric_fieldname[0]

        logger.info(f'Using the best metric from CSV field `{best_metric_fieldname}`')
        load_epoch = int(exp.get_best_model_stat(best_metric_fieldname, best_metric_is_better)['epoch'])
    elif args.load_epoch is None:
        load_epoch = None
    elif args.load_epoch >= 0:
        exp.load_summary()
        load_epoch = args.load_epoch
    else:
        raise ValueError(f"Wrong args.load_epoch value: {args.load_epoch}")

    if load_epoch is not None:
        predictions_file_path = os.path.join(exp.predictions_dir, f'epoch_{load_epoch:04d}_{args.mode}val.pkl')
    else:
        predictions_file_path = os.path.join(exp.predictions_dir, f'pretrained_{args.mode}val.pkl')

    with open(predictions_file_path, 'rb') as f:
        predictions = pickle.load(f)

    video_predictions = predictions['video_predictions']
    video_labels = predictions['video_labels']
    #video_ids = predictions['video_ids']

    output_dir = os.path.join(exp.plots_dir, 'per_class_accuracy')

    tb_writer = SummaryWriter(os.path.join(exp.tensorboard_runs_dir, 'val_confusion_matrix'), comment='val_confusion_matrix')

    for shrink in [False, True]:
        if args.sort_method == 'all':
            for sort_method in ['train_class_frequency', 'val_class_frequency', 'val_per_class_accuracy']:
                logger.info(sort_method)
                generate_confusion_matrix(args, sort_method, cfg.dataset_cfg, output_dir, video_predictions, video_labels, tb_writer, shrink)
        else:
            generate_confusion_matrix(args, args.sort_method, cfg.dataset_cfg, output_dir, video_predictions, video_labels, tb_writer, shrink)



if __name__ == '__main__':
    main()
