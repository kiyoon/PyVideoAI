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

    dataset_cfg = dataset_configs.load_cfg(args.dataset)

    exp = ExperimentBuilder(args.experiment_root, args.dataset, args.model, args.experiment_name)
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

    pred_labels = np.argmax(video_predictions, axis=1)
    pred_labels2 = np.argmax(video_predictions2, axis=1)
    '''
    for i in range(10):
        print(video_predictions[i].max())
        print(video_predictions[i].min())
        print(video_labels[i])
        print()
    '''

    class_indices = range(dataset_cfg.num_classes)
    class_keys = dataset_cfg.class_keys 

    def grouped_accuracy(pred_labels, video_labels):
        num_total_val_samples = {}
        num_correct_pred = {}
        for pred, label in zip(pred_labels, video_labels):
            class_key = class_keys[label]
            first_word = class_key.split(' ', 1)[0]

            if pred == label:
                if first_word in num_correct_pred.keys():
                    num_correct_pred[first_word] += 1
                    num_total_val_samples[first_word] += 1
                else:
                    num_correct_pred[first_word] = 1
                    num_total_val_samples[first_word] = 1
            else:
                if first_word in num_correct_pred.keys():
                    num_total_val_samples[first_word] += 1
                else:
                    num_correct_pred[first_word] = 0
                    num_total_val_samples[first_word] = 1

        return num_correct_pred, num_total_val_samples

    num_correct_pred, num_total_val_samples = grouped_accuracy(pred_labels, video_labels)
    num_correct_pred2, num_total_val_samples2 = grouped_accuracy(pred_labels2, video_labels2)

    diff_correct_pred = {}
    diff_accuracy = {}
    for first_word in num_correct_pred.keys():
        diff_correct_pred[first_word] = num_correct_pred2[first_word] - num_correct_pred[first_word]
        diff_accuracy[first_word] = (num_correct_pred2[first_word] - num_correct_pred[first_word]) / num_total_val_samples[first_word]

    index = np.arange(len(diff_correct_pred.keys()))

    #sort_index = np.fromiter(diff_correct_pred.values(), dtype=np.int).argsort()[::-1]
    sort_index = np.fromiter(diff_accuracy.values(), dtype=np.float).argsort()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.barh(index, np.fromiter(diff_correct_pred.values(), dtype=np.int)[sort_index], height=1, color='b')
    #ax.bar(index, diff_correct_pred.values(), width=1, color='b')
    #plt.ylabel('Class Group', fontsize=12)
    plt.xlabel('Correct sample count difference', fontsize=12)
    plt.yticks(index,  np.array(list(num_correct_pred.keys()))[sort_index], fontsize=6, rotation=0) 
    plt.xticks(fontsize=12, rotation=0) 

    fig.set_tight_layout(True)

    
    # Generate visualisation and summary files

    out_dir = os.path.join(args.output_dir, 'grouped_accuracy_comparison', args.dataset, '{:s}_{:s}_{:s}'.format(args.model, args.experiment_name, args.mode), '{:s}_{:s}_{:s}'.format(args.model2, args.experiment_name2, args.mode2), 'sort_%s' % sort_method)
    os.makedirs(out_dir, exist_ok = True)

    plt.savefig('%s/diff.pdf' % (out_dir))
    plt.savefig('%s/diff.png' % (out_dir))


    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.barh(index, np.fromiter(diff_accuracy.values(), dtype=np.float)[sort_index], height=1, color='b')
    #plt.ylabel('Class Group', fontsize=12)
    plt.xlabel('Class accuracy difference', fontsize=12)
    plt.yticks(index, np.array(list(num_correct_pred.keys()))[sort_index], fontsize=6, rotation=0) 
    plt.xticks(fontsize=12, rotation=0) 

    fig.set_tight_layout(True)

    
    # Generate visualisation and summary files

    plt.savefig('%s/normalised.pdf' % (out_dir))
    plt.savefig('%s/normalised.png' % (out_dir))

    with open('%s/details.csv' % (out_dir), mode='w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=str(','), quotechar=str('"'), quoting=csv.QUOTE_MINIMAL)

        csvwriter.writerow(['class_key_first_word', 'accuracy1 (%)', 'accuracy2 (%)', 'accuracy_diff', 'num_correct_pred1', 'num_correct_pred2', 'num_correct_diff', 'num_samples_in_val'])

        for key in num_correct_pred.keys():
            
            accuracy1 = float(num_correct_pred[key]) / num_total_val_samples[key] * 100
            accuracy2 = float(num_correct_pred2[key]) / num_total_val_samples2[key] * 100
            accuracy_diff = accuracy2-accuracy1

            csvwriter.writerow([key, accuracy1, accuracy2, accuracy_diff, num_correct_pred[key], num_correct_pred2[key], num_correct_pred2[key]-num_correct_pred[key], num_total_val_samples[key]])

def main():
    parser = argparse.ArgumentParser(
        description='Load predictions, and generating per-class accuracy and confusion matrix')

    add_exp_arguments(parser, dataset_configs.available_datasets, model_configs.available_models, root_default=config.DEFAULT_EXPERIMENT_ROOT, dataset_default='epic_verb', model_default='i3d', name_default='test')
    parser.add_argument("-o", "--output_dir", type=str, default="visualisations",  help="output directory")
    parser.add_argument("-l", "--load_epoch", type=int, default=-2, help="Load from checkpoint. Set to -1 to load from the last checkpoint, and to -2 to load best model in terms of val_acc.")
    parser.add_argument("-L", "--load_epoch2", type=int, default=-2, help="Load from checkpoint. Set to -1 to load from the last checkpoint, and to -2 to load best model in terms of val_acc.")
    parser.add_argument("-m", "--mode", type=str, default="oneclip", choices=["oneclip", "multicrop"],  help="Mode used for run_val.py")
    parser.add_argument("-2", "--mode2", type=str, default="oneclip", choices=["oneclip", "multicrop"],  help="Mode used for run_val.py")

    parser.add_argument("-O", "--model2", type=str, default="i3d", choices=model_configs.available_models,  help="Mode used for run_val.py")
    parser.add_argument("-X", "--experiment_name2", type=str, default="test",  help="")

    parser.add_argument(
        '--sort_method', type=str, default='all', choices=['train_class_frequency', 'val_class_frequency', 'val_per_class_accuracy', 'all'], help='Sorting method')

    args = parser.parse_args()

    generate_confusion_matrix(args, args.sort_method, 0)



if __name__ == '__main__':
    main()
