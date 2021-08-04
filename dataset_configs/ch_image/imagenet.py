import os
import numpy as np
import pandas as pd

from video_datasets_api.imagenet.definitions import NUM_CLASSES as num_classes
from video_datasets_api.imagenet.read_annotations import parse_meta_mat 
from pyvideoai.config import DATA_DIR

from pyvideoai.tasks import SingleLabelClassificationTask
task = SingleLabelClassificationTask()

# Paths
dataset_root = os.path.join(DATA_DIR, 'ILSVRC2012')
train_dir = os.path.join(dataset_root, 'images', 'train')
val_dir = os.path.join(dataset_root, 'images', 'val')
split_file_dir = os.path.join(dataset_root, "splits")
split_file_basename = {'train': 'train.txt', 'val': 'val.txt', 'multicropval': 'val.txt'}
split2mode = {'train': 'train', 'val': 'test', 'multicropval': 'test', 'test': 'test'}

meta_mat_path = os.path.join(dataset_root, 'ILSVRC2012_devkit_t12/data/meta.mat')
_, wnid_to_label, class_keys = parse_meta_mat(meta_mat_path)
class_keys = pd.DataFrame(class_keys, columns=['class_keys'])['class_keys']

# Misc

from pyvideoai.dataloaders.image_classification_dataset import count_class_frequency
def count_train_class_frequency():
    """Count class frequency in training data.
    Used for generating confusion matrix. (generate_confusion_matrix.py)
    """
    return count_class_frequency(os.path.join(split_file_dir, split_file_basename['train']))


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
