import os
import numpy as np
import pandas as pd

from video_datasets_api.something_something_v2.definitions import NUM_CLASSES as num_classes
from video_datasets_api.something_something_v2.read_annotations import get_class_keys_shrinked
from pyvideoai.config import DATA_DIR

from pyvideoai.tasks import SingleLabelClassificationTask
task = SingleLabelClassificationTask()

# Paths
dataset_root = os.path.join(DATA_DIR, 'something-something-v2')
frames_dir = '/dev/shm/datasets/something-something-v2/frames_q5'
annotations_root = os.path.join(dataset_root, 'annotations')
label_json = os.path.join(annotations_root, 'something-something-v2-labels.json')
class_keys = pd.DataFrame(get_class_keys_shrinked(label_json), columns=['class_keys'])['class_keys']

video_split_file_dir = os.path.join(dataset_root, "splits_video")
frames_split_file_dir = os.path.join(dataset_root, "splits_frames")
split_file_basename = {'train': 'train.csv', 'val': 'val.csv', 'multicropval': 'val.csv'}
split2mode = {'train': 'train', 'val': 'test', 'multicropval': 'test', 'test': 'test'}


# Misc

from pyvideoai.dataloaders.frames_sparsesample_dataset import count_class_frequency as count_cf
def count_class_frequency(split='train'):
    """Count class frequency in training data.
    Used for generating confusion matrix. (generate_confusion_matrix.py)
    """
    return count_cf(os.path.join(frames_split_file_dir, split_file_basename[split]))
    


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
        fig = plt.figure(figsize = (16,10))
        ax = fig.add_subplot(111)

        sn.set(font_scale=10)#for label size
        sn_plot = sn.heatmap(df_confusion_matrix, annot=False, annot_kws={"size": 12}, cbar_kws={"shrink": 0.5}, cmap=cmap, square=True, xticklabels=False, yticklabels=1, vmin=vmin, vmax=vmax, center=center)# font size
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
