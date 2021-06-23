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
import time

from pyvideoai import config, dataset_configs, model_configs

import json
from video_datasets_api.cater.generate_labels_from_scenes import class_keys_task2, class_keys_to_labels, generate_task2_labels_from_scenes



def main():
    parser = argparse.ArgumentParser(
        description='Load predictions, and generating per-class accuracy and confusion matrix')

    add_exp_arguments(parser, dataset_configs.available_datasets, model_configs.available_models, root_default=config.DEFAULT_EXPERIMENT_ROOT, dataset_default='cater_task2', model_default='trn_resnet50', name_default='lr01_smallscalejitter')
    parser.add_argument("-s", "--scenes_dir", type=str, default="data/cater/max2action/scenes",  help="Where to read scenes data")
    parser.add_argument("-l", "--load_epoch", type=int, default=-2, help="Load from checkpoint. Set to -1 to load from the last checkpoint, and to -2 to load best model in terms of val_acc.")
    parser.add_argument("-m", "--mode", type=str, default="oneclip", choices=["oneclip", "multicrop"],  help="Mode used for run_val.py")

    args = parser.parse_args()


    dataset_cfg = dataset_configs.load_cfg(args.dataset)

    perform_multicropval=True       # when loading, assume there was multicropval. Even if there was not, having more CSV field information doesn't hurt.
    assert dataset_cfg.task == 'multilabel_classification'
    summary_fieldnames, summary_fieldtypes = ExperimentBuilder.return_fields_multilabel(multicropval = perform_multicropval)
    exp = ExperimentBuilder(args.experiment_root, args.dataset, args.model, args.experiment_name, summary_fieldnames = summary_fieldnames, summary_fieldtypes = summary_fieldtypes, telegram_key_ini = config.KEY_INI_PATH)
    exp.load_summary()

    if args.load_epoch == -1:
        load_epoch = int(exp.summary['epoch'][-1])
    elif args.load_epoch == -2:
        if args.mode == 'oneclip':
            field = 'val_vid_mAP'
        else:   # multicrop
            field = 'multi_crop_val_vid_mAP'
        load_epoch = int(exp.get_best_model_stat(field)['epoch'])
    elif args.load_epoch >= 0:
        load_epoch = args.load_epoch
    else:
        raise ValueError("Wrong args.load_epoch value: {:d}".format(args.load_epoch))

    predictions_file_path = os.path.join(exp.predictions_dir, 'epoch_%04d_%sval.pkl' % (load_epoch, args.mode))
    with open(predictions_file_path, 'rb') as f:
        predictions = pickle.load(f)

    video_predictions = predictions['video_predictions']
    video_labels = predictions['video_labels']
    video_ids = predictions['video_ids']

    MAX_ACTION_DUR = 30

    # 331 elements
    histogram_x = np.arange(-MAX_ACTION_DUR, dataset_cfg.num_classes, dtype=np.int)        # -30 to 300
    accuracy_histogram = np.zeros(dataset_cfg.num_classes + MAX_ACTION_DUR, dtype=np.float)
    count_histogram = np.zeros(dataset_cfg.num_classes + MAX_ACTION_DUR, dtype=np.int)

    classes = class_keys_task2(string=False, only_before=True)
    class_keys_to_labels_dict = class_keys_to_labels(classes)
    for video_id, video_label, video_prediction in zip(video_ids, video_labels, video_predictions):
        scene_path = os.path.join(args.scenes_dir, f'CATER_new_{video_id:06d}.json')
        with open(scene_path, 'r', encoding='utf8') as scene_file:
            scene = json.load(scene_file)

        labels_scenes, actions_in_charge = generate_task2_labels_from_scenes(scene['movements'], scene['objects'], class_keys_to_labels_dict)
        video_label_no_onehot = np.where(video_label==1)[0]
        assert set(video_label_no_onehot) == labels_scenes, f"Labels generated does not match with labels provided in `lists`. {video_label_no_onehot} and {sorted(list(labels_scenes))}"

        for label in labels_scenes:
            order = classes[label][1][0]    # before or during
            for action_pair in actions_in_charge[label]:
                if order == 'before':
                    distance = action_pair[1].start_time() - action_pair[0].end_time()
                elif order == 'during':
                    distance = max(action_pair[0].start_time(), action_pair[1].start_time()) - min(action_pair[0].end_time(), action_pair[1].end_time())
                else:
                    raise NotImplementedError()
                accuracy_histogram[distance+MAX_ACTION_DUR] += video_prediction[label]
                count_histogram[distance+MAX_ACTION_DUR] += 1

    for i, (accuracy, count) in enumerate(zip(accuracy_histogram, count_histogram)):
        if count > 0:
            accuracy_histogram[i] = accuracy / count

    fig, ax1 = plt.subplots(figsize=(12,7))
    color = 'tab:red'
    #ax1.set_xticks(histogram_x)
    #ax1.set_xticklabels(class_key_contains, rotation=xtickrotation)
    #ax1.set_ylim([0,1])
    ax1.set_ylabel('Accuracy', color=color)
    ax1.plot(histogram_x, accuracy_histogram, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('# samples (x)', color=color)  # we already handled the x-label with ax1
    #ax2.set_ylim([0,200])
    ax2.scatter(histogram_x, count_histogram, color=color, marker='x')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title(f'Accuracy vs distance of action pairs')

    pdfpath = os.path.join(exp.plots_dir, f"accuracy_vs_distance-epoch_{load_epoch:04d}_{args.mode}val.pdf")
    print(f"Saving to {pdfpath}")
    plt.savefig(pdfpath)

    pngpath = os.path.join(exp.plots_dir, f"accuracy_vs_distance-epoch_{load_epoch:04d}_{args.mode}val.png")
    print(f"Saving to {pngpath}")
    plt.savefig(pngpath)

if __name__ == '__main__':
    main()
