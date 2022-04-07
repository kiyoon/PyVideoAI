# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""
Build confusion matrix
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

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
from utils.video_accuracy import VideoAccuracy
import dataset_configs
import model_configs
import time

import config

vmin = -0.3
vmax = 0.3


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



def generate_confusion_matrix(args, sort_method, shrink=0):
    """save confusion matrix
    Params:
        shrink (int): set to 0 if you don't want to shrink the matrix. n for the first n, -n for the last n.

    """
    print(sort_method)
    print(shrink)

    dataset_cfg = dataset_configs.load_cfg(args.dataset)

    exp = ExperimentBuilder(args.experiment_root, args.dataset, args.model, args.experiment_name, args.subfolder_name)
    exp.load_summary()

    def load_predictions(load_epoch, mode, exp):
        real_load_epoch = None
        if load_epoch == -1:
            real_load_epoch = int(exp.summary['epoch'][-1])
        elif load_epoch == -2:
            if mode == 'oneclip':
                field = 'val_acc'
            else:   # multicrop
                field = 'multi_crop_val_vid_acc_top1'
            real_load_epoch = int(exp.get_best_model_stat(field)['epoch'])
        elif load_epoch >= 0:
            real_load_epoch = load_epoch
        else:
            raise ValueError("Wrong load_epoch value: {:d}".format(load_epoch))

        predictions_file_path = os.path.join(exp.predictions_dir, 'epoch_%04d_%sval.pkl' % (real_load_epoch, mode))
        with open(predictions_file_path, 'rb') as f:
            predictions = pickle.load(f)

        video_predictions = predictions['video_predictions']
        video_labels = predictions['video_labels']
        #video_ids = predictions['video_ids']

        return video_predictions, video_labels

    video_predictions, video_labels = load_predictions(args.load_epoch, args.mode, exp)


    ## 2
    exp2 = ExperimentBuilder(args.experiment_root, args.dataset, args.model2, args.experiment_name2)
    exp2.load_summary()

    video_predictions2, video_labels2 = load_predictions(args.load_epoch2, args.mode2, exp2)

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
    cm1 = confusion_matrix(video_labels, pred_labels, labels = class_indices)

    pred_labels2 = np.argmax(video_predictions2, axis=1)
    cm2 = confusion_matrix(video_labels2, pred_labels2, labels = class_indices)

    num_samples_per_target = cm1.sum(axis=1)
    cm1 = normalize(cm1, axis=1, norm='l1')     # row (true labels) will sum to 1.
    cm2 = normalize(cm2, axis=1, norm='l1')     # row (true labels) will sum to 1.
    cm = cm2-cm1

    class_frequency_in_train = dataset_cfg.count_train_class_frequency()

    if sort_method == 'val_class_frequency':
        # sort by class frequency in validation data (video-based)
        sort_labels = num_samples_per_target.argsort()[::-1]

    elif sort_method == 'train_class_frequency':
        # sort by class frequency in training data (video-based)
        sort_labels = class_frequency_in_train.argsort()[::-1]

    elif sort_method == 'val_per_class_accuracy':
        # sort by accuracy per class (video-based)
        sort_labels = cm.diagonal().argsort()[::-1]

    else:
        raise ValueError('Wrong sort_method')


    if shrink != 0:
        # remove the ones with too few val samples
        # whilst keeping the top n classes

        if shrink < 0:
            sort_labels = sort_labels[::-1]
        labels_to_keep = []

        if args.dataset == 'something_v1':
            class_category_count = {}

        for i in sort_labels:
            # for sthsth, only plot classes with two [something]
            #if class_keys[i].count('[]') >= 2:
            #    labels_to_keep.append(i)

            if num_samples_per_target[i] >= 20:
                labels_to_keep.append(i)
                if args.dataset == 'something_v1':
                    first_word = class_keys[i].split(' ', 1)[0]
                    if first_word in class_category_count.keys():
                        class_category_count[first_word] += 1
                    else:
                        class_category_count[first_word] = 1


            if len(labels_to_keep) >= abs(shrink):
                break


        if args.dataset == 'something_v1':
            print(class_category_count)

        _, pred_labels, video_labels, _, _ = shrink_labels(dataset_cfg.num_classes, labels_to_keep, class_keys, pred_labels, video_labels, sort_labels, class_frequency_in_train)
        class_keys, pred_labels2, video_labels2, sort_labels, class_frequency_in_train = shrink_labels(dataset_cfg.num_classes, labels_to_keep, class_keys, pred_labels2, video_labels2, sort_labels, class_frequency_in_train)
        ##



    cm1_sorted = confusion_matrix(video_labels, pred_labels, labels = sort_labels)
    cm2_sorted = confusion_matrix(video_labels2, pred_labels2, labels = sort_labels)

    num_samples_per_target = cm1_sorted.sum(axis=1)  # class frequency in val
    num_correct_pred_per_target1 = cm1_sorted.diagonal()
    num_correct_pred_per_target2 = cm2_sorted.diagonal()

    cm1_sorted = normalize(cm1_sorted, axis=1, norm='l1')     # row (true labels) will sum to 1.
    cm2_sorted = normalize(cm2_sorted, axis=1, norm='l1')     # row (true labels) will sum to 1.
    cm_sorted = cm2_sorted-cm1_sorted

    class_frequency_in_train = class_frequency_in_train[sort_labels]


    # Generate visualisation and summary files

    out_dir = os.path.join(args.output_dir, 'confusion_comparison', args.dataset, '{:s}_{:s}_{:s}'.format(args.model, args.experiment_name, args.mode), '{:s}_{:s}_{:s}'.format(args.model2, args.experiment_name2, args.mode2), 'sort_%s' % sort_method)
    os.makedirs(out_dir, exist_ok = True)

    df_cm = pd.DataFrame(cm_sorted, class_keys[sort_labels],
                              class_keys[sort_labels])

    assert cm_sorted.diagonal().max() < vmax and cm_sorted.diagonal().min() > vmin
    fig = dataset_cfg.plot_confusion_matrix(df_cm, shrink=shrink!=0, vmin=vmin, vmax=vmax, cmap=None, center=0)
    shrink_str = 'shrinked_{:d}_'.format(shrink) if shrink!=0 else ''

    plt.savefig('%s/%sconfusion.pdf' % (out_dir, shrink_str))
    plt.savefig('%s/%sconfusion.png' % (out_dir, shrink_str))

    with open('%s/%sper_class_accuracy.csv' % (out_dir, shrink_str), mode='w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=str(','), quotechar=str('"'), quoting=csv.QUOTE_MINIMAL)

        csvwriter.writerow(['class_key', 'accuracy1 (%)', 'accuracy2 (%)', 'accuracy_diff', 'num_correct_pred1', 'num_correct_pred2', 'num_correct_diff', 'num_samples_in_val', 'num_samples_in_train'])

        for class_label, num_correct_pred, num_correct_pred2, num_samples_in_val, num_samples_in_train in zip(class_keys[sort_labels], num_correct_pred_per_target1, num_correct_pred_per_target2, num_samples_per_target, class_frequency_in_train):
            accuracy1 = float(num_correct_pred) / num_samples_in_val * 100 if num_samples_in_val != 0 else 'NaN'
            accuracy2 = float(num_correct_pred2) / num_samples_in_val * 100 if num_samples_in_val != 0 else 'NaN'
            if accuracy1 == 'NaN' or accuracy2 == 'NaN':
                accuracy_diff = 'NaN'
            else:
                accuracy_diff = accuracy2-accuracy1

            csvwriter.writerow([class_label, accuracy1, accuracy2, accuracy_diff, num_correct_pred, num_correct_pred2, num_correct_pred2-num_correct_pred, num_samples_in_val, num_samples_in_train])


def main():
    parser = argparse.ArgumentParser(
        description='Load predictions, and generating per-class accuracy and confusion matrix')

    add_exp_arguments(parser, dataset_configs.available_datasets, model_configs.available_models, root_default=config.DEFAULT_EXPERIMENT_ROOT, dataset_default='epic_verb', model_default='i3d', name_default='test')
    parser.add_argument("-o", "--output_dir", type=str, default="visualisations",  help="output directory")
    parser.add_argument("-l", "--load_epoch", type=int, default=-2, help="Load from checkpoint. Set to -1 to load from the last checkpoint, and to -2 to load best model in terms of val_acc.")
    parser.add_argument("-L", "--load_epoch2", type=int, default=-2, help="Load from checkpoint. Set to -1 to load from the last checkpoint, and to -2 to load best model in terms of val_acc.")
    parser.add_argument("-m", "--mode", type=str, default="oneclip", choices=["oneclip", "multicrop"],  help="Mode used for run_val.py")
    parser.add_argument("-m2", "--mode2", type=str, default="oneclip", choices=["oneclip", "multicrop"],  help="Mode used for run_val.py")

    parser.add_argument("-M2", "--model2", type=str, default="i3d", choices=model_configs.available_models,  help="The second model")
    parser.add_argument("-E2", "--experiment_name2", type=str, default="test",  help="The second experiment name")
    parser.add_argument("-S2", "--subfolder_name2", type=str, default=None,  help="The second experiment's subfolder name")

    parser.add_argument(
        '--sort_method', type=str, default='all', choices=['train_class_frequency', 'val_class_frequency', 'val_per_class_accuracy', 'all'], help='Sorting method')

    args = parser.parse_args()

    #for shrink in [0, 10, -10, 20, -20]:
    for shrink in [30, -30]:
        if args.sort_method == 'all':
            for sort_method in ['train_class_frequency', 'val_class_frequency', 'val_per_class_accuracy']:
                generate_confusion_matrix(args, sort_method, shrink)
        else:
            generate_confusion_matrix(args, args.sort_method, shrink)



if __name__ == '__main__':
    main()
