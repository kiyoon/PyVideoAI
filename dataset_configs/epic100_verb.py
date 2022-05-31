import os
#import pandas as pd
import pickle
import numpy as np

from video_datasets_api.epic_kitchens_100.definitions import NUM_VERB_CLASSES as num_classes
from video_datasets_api.epic_kitchens_100.read_annotations import get_verb_uid2label_dict, epic_narration_id_to_unique_id
from video_datasets_api.epic_kitchens_100.get_class_keys import EPIC100_get_class_keys
from video_datasets_api.epic_kitchens_100.get_head_tail_classes import get_head_tail_classes

from pyvideoai.config import DATA_DIR

from pyvideoai.tasks import SingleLabelClassificationTask
task = SingleLabelClassificationTask()


# Paths
dataset_root = os.path.join(DATA_DIR, 'EPIC_KITCHENS_100')
video_dir = os.path.join(dataset_root, 'segments324_15fps')
flowframes_dir = os.path.join(dataset_root, 'flow_frames')
gulp_rgb_dirname = {'train': 'gulp_rgb/train', 'val': 'gulp_rgb/val', 'multicropval': 'gulp_rgb/val'}
gulp_flow_dirname = {'train': 'gulp_flow/train', 'val': 'gulp_flow/val', 'multicropval': 'gulp_flow/val', 'trainpartialdata_testmode': 'gulp_flow/train', 'traindata_testmode': 'gulp_flow/train'}
annotations_root = os.path.join(dataset_root, 'epic-kitchens-100-annotations')

narration_id_to_video_id, narration_id_sorted = epic_narration_id_to_unique_id(annotations_root)
uid2label = get_verb_uid2label_dict(annotations_root, narration_id_to_video_id)
class_keys = EPIC100_get_class_keys(annotations_root, 'verb')
class_keys_to_label_idx = {}
for label_idx, class_key in enumerate(class_keys):
    class_keys_to_label_idx[class_key] = label_idx
head_classes, tail_classes = get_head_tail_classes(annotations_root, 'verb')

video_split_file_dir = os.path.join(dataset_root, "splits_video")
frames_split_file_dir = os.path.join(dataset_root, "splits_frames")
flowframes_split_file_dir = os.path.join(dataset_root, "splits_flowframes")
gulp_rgb_split_file_dir = os.path.join(dataset_root, 'splits_gulp_rgb')
gulp_flow_split_file_dir = os.path.join(dataset_root, 'splits_gulp_flow')
split_file_basename = {'train': 'train.csv', 'val': 'val.csv', 'multicropval': 'val.csv', 'traindata_testmode': 'train.csv', 'trainpartialdata_testmode': 'train_partial.csv', 'multicropval_strided': 'val.csv', 'multicroptrain_strided': 'train.csv'}
split2mode = {'train': 'train', 'val': 'test', 'multicropval': 'test', 'test': 'test', 'traindata_testmode': 'test', 'trainpartialdata_testmode': 'test', 'multicropval_strided': 'test', 'multicroptrain_strided': 'test'}
horizontal_flip = True



# Misc

def count_train_class_frequency():
    """Count class frequency in training data.
    Used for generating confusion matrix. (generate_confusion_matrix.py)
    """
    with open(os.path.join(annotations_root, 'EPIC_100_train.pkl'), 'rb') as f:
        df = pickle.load(f)

    class_frequency = np.bincount(df['verb_class'], minlength=num_classes)
    return class_frequency

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
