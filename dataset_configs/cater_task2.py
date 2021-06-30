import os
import numpy as np
import pandas as pd

from video_datasets_api.cater.definitions import NUM_CLASSES_TASK2 as num_classes
from video_datasets_api.cater.class_keys import class_keys_task2
from pyvideoai.config import DATA_DIR

from pyvideoai.tasks import MultiLabelClassificationTask
task = MultiLabelClassificationTask()

# Paths
dataset_root = os.path.join(DATA_DIR, "cater/max2action")
frames_dir = os.path.join(dataset_root, 'frames_q5')
frames_split_file_dir = os.path.join(dataset_root, "splits_frames/actions_order_uniq")
split_file_basename = {'train': 'train.txt', 'val': 'val.txt', 'multicropval': 'val.txt'}
split2mode = {'train': 'train', 'val': 'test', 'multicropval': 'test', 'test': 'test'}

class_keys = class_keys_task2()


# Misc

def count_train_class_frequency():
    """Count class frequency in training data.
    Used for generating confusion matrix. (generate_confusion_matrix.py)
    """
    return np.ones((num_classes, 70), dtype=np.int)


'''
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
'''
