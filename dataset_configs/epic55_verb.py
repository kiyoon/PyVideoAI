import os
#import pandas as pd
import pickle
import numpy as np

from video_datasets_api.epic_kitchens_55.definitions import NUM_CLASSES as num_classes
from video_datasets_api.epic_kitchens_55.read_annotations import get_verb_uid2label_dict
from video_datasets_api.epic_kitchens_55.epic_get_verb_class_keys import EPIC_get_verb_class_keys
from pyvideoai.config import DATA_DIR

from pyvideoai.tasks import SingleLabelClassificationTask
task = SingleLabelClassificationTask()


# Paths
dataset_root = os.path.join(DATA_DIR, 'EPIC_KITCHENS_55')
video_dir = os.path.join(dataset_root, 'segments_15fps_324')
annotations_root = os.path.join(dataset_root, 'epic-kitchens-55-annotations')
train_action_labels_pkl = os.path.join(annotations_root, 'EPIC_train_action_labels.pkl')
#detection_frame_root = '/disk/scratch_fast1/s1884147/datasets/epic_detection_frame_extracted'


TRAINING_DATA = ['P' + x.zfill(2) for x in map(str, range(1, 26))]  # P01 to P25

uid2label = get_verb_uid2label_dict(train_action_labels_pkl)
class_keys = EPIC_get_verb_class_keys(os.path.join(annotations_root, 'EPIC_verb_classes.csv'))

video_split_file_dir = os.path.join(dataset_root, "splits_video")
frames_split_file_dir = os.path.join(dataset_root, "splits_frames")
split_file_basename = {'train': 'train.csv', 'val': 'val.csv', 'multicropval': 'val.csv'}
split2mode = {'train': 'train', 'val': 'test', 'multicropval': 'test', 'test': 'test'}

# Training settings
horizontal_flip = True 



# Misc

def count_train_class_frequency():
    """Count class frequency in training data.
    Used for generating confusion matrix. (generate_confusion_matrix.py)
    """
    with open(train_action_labels_pkl, 'rb') as f:
        df = pickle.load(f)

    training_mask = np.isin(df['participant_id'], TRAINING_DATA)
    class_frequency = np.bincount(df['verb_class'][training_mask], minlength=num_classes)
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
