############
"""
The dataset_config needs:

For PyVideoAI core:
(required)      num_classes (int)
                task (pyvideoai.tasks.task)
(suggested)     class_keys (list of str) - for visualisation
(optional)      count_train_class_frequency (callable) - for confusion matrix plotting
                plot_confusion_matrix (callable) - for confusion matrix plotting


(Below are optional, but better to keep the format consistent
so that the experiment_config doesn't need to change much between datasets.)

It's likely that your exp_config uses:
    dataset_root (str)
    frame_dir (str)
    frames_split_file_dir (str)
    split_file_basename (dict)
    split2mode (dict)
"""

import os
import numpy as np
import pandas as pd

from video_datasets_api.hmdb.definitions import NUM_CLASSES as num_classes
from video_datasets_api.hmdb.read_annotations import get_class_keys
from pyvideoai.config import DATA_DIR

from pyvideoai.tasks import SingleLabelClassificationTask
task = SingleLabelClassificationTask()

# Paths
class_keys = pd.DataFrame(get_class_keys(), columns=['class_keys'])['class_keys']


"""
Optional but suggested to keep the format consistent.
"""
dataset_root = os.path.join(DATA_DIR, 'hmdb51')
frames_dir = os.path.join(dataset_root, "frames_q5")
gulp_rgb_dirname = {'train': 'gulp_rgb', 'val': 'gulp_rgb', 'multicropval': 'gulp_rgb', 'trainpartialdata_testmode': 'gulp_flow', 'traindata_testmode': 'gulp_flow'}
gulp_flow_dirname = {'train': 'gulp_flow', 'val': 'gulp_flow', 'multicropval': 'gulp_flow', 'trainpartialdata_testmode': 'gulp_flow', 'traindata_testmode': 'gulp_flow'}
frames_split_file_dir = os.path.join(dataset_root, "splits_frames")
gulp_rgb_split_file_dir = os.path.join(dataset_root, 'splits_gulp_rgb')
gulp_flow_split_file_dir = os.path.join(dataset_root, 'splits_gulp_flow')
split_file_basename1 = {'train': 'train1.csv', 'val': 'test1.csv', 'multicropval': 'test1.csv', 'traindata_testmode': 'train1.csv'}
split_file_basename2 = {'train': 'train2.csv', 'val': 'test2.csv', 'multicropval': 'test2.csv', 'traindata_testmode': 'train2.csv'}
split_file_basename3 = {'train': 'train3.csv', 'val': 'test3.csv', 'multicropval': 'test3.csv', 'traindata_testmode': 'train3.csv'}
split_file_basename = split_file_basename1
split2mode = {'train': 'train', 'val': 'test', 'multicropval': 'test', 'test': 'test', 'traindata_testmode': 'test'}

# Training settings
horizontal_flip = True

# Misc

def count_train_class_frequency():
    """Count class frequency in training data.
    Used for generating confusion matrix. (generate_confusion_matrix.py)
    HMDB is completely balanced and there's no need to do this
    """
    return np.ones(num_classes, dtype=np.int) * 70


import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import seaborn as sn
def plot_confusion_matrix(df_confusion_matrix, shrink=False, vmin=0, vmax=1, cmap='YlGnBu', center=None):
    if not shrink:
        fig = plt.figure(figsize = (65,50))
        ax = fig.add_subplot(111)
        # x label on top
        ax.xaxis.tick_top()

        sn.set(font_scale=10)#for label size
        sn_plot = sn.heatmap(df_confusion_matrix, annot=False, annot_kws={"size": 12}, cmap=cmap, square=True, xticklabels=1, yticklabels=1, vmin=vmin, vmax=vmax, center=center)# font size
        plt.xlabel('Predicted', fontsize=50)
        plt.ylabel('Target', fontsize=50)

        # This sets the yticks "upright" with 0, as opposed to sideways with 90.
        plt.yticks(fontsize=12, rotation=0)
        plt.xticks(fontsize=12, rotation=90)

        # here set the colorbar labelsize by 50
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=50)
    else:
        #fig = plt.figure(figsize = (23,20))
        fig = plt.figure(figsize = (10,10))
        ax = fig.add_subplot(111)
        # x label on top
        ax.xaxis.tick_top()

        sn.set(font_scale=10)#for label size
        sn_plot = sn.heatmap(df_confusion_matrix, annot=False, annot_kws={"size": 12}, cbar_kws={"shrink": 0.5}, cmap=cmap, square=True, xticklabels=1, yticklabels=1, vmin=vmin, vmax=vmax, center=center)# font size
        plt.xlabel('Predicted', fontsize=20)
        plt.ylabel('Target', fontsize=20)

        # This sets the yticks "upright" with 0, as opposed to sideways with 90.
        plt.yticks(fontsize=12, rotation=0)
        plt.xticks(fontsize=12, rotation=90)

        # here set the colorbar labelsize by 20
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=20)
    fig.set_tight_layout(True)

    return fig
