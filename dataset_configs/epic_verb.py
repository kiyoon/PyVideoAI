import os
#import pandas as pd
import pickle
import numpy as np

from video_datasets_api.epic_kitchens.definitions import NUM_CLASSES as num_classes
from video_datasets_api.epic_kitchens.read_annotations import get_verb_uid2label_dict
from video_datasets_api.epic_kitchens.epic_get_verb_class_keys import EPIC_get_verb_class_keys
from pyvideoai.config import DATA_DIR


task = 'singlelabel_classification'


# Paths
dataset_root = os.path.join(DATA_DIR, 'EPIC_KITCHENS_2018')
annotations_root = os.path.join(dataset_root, 'annotations')
train_action_labels_pkl = os.path.join(annotations_root, 'EPIC_train_action_labels.pkl')
#detection_frame_root = '/disk/scratch_fast1/s1884147/datasets/epic_detection_frame_extracted'


TRAINING_DATA = ['P' + x.zfill(2) for x in map(str, range(1, 26))]  # P01 to P25

uid2label = get_verb_uid2label_dict(train_action_labels_pkl)
class_keys = EPIC_get_verb_class_keys(os.path.join(annotations_root, 'EPIC_verb_classes.csv'))

#video_split_file_dir = "/home/kiyoon/datasets/EPIC_KITCHENS_2018/epic_list"
video_split_file_dir = os.path.join(dataset_root, "epic_list")
frames_split_file_dir = os.path.join(dataset_root, "epic_list_frames")
split_file_basename = {'train': 'train.csv', 'val': 'val.csv', 'multicropval': 'val.csv'}
split2mode = {'train': 'train', 'val': 'val', 'multicropval': 'test', 'test': 'test'}

#def _unpack_data_videoclip(data):
#    inputs, uids, labels, spatial_temporal_idx, _, _ = data
#    return _data_to_gpu(inputs, uids, labels) + (spatial_temporal_idx,)
#
#def _unpack_data_sparse(data):
#    inputs, uids, labels, spatial_idx, _, _, _ = data
#    return _data_to_gpu(inputs, uids, labels) + (spatial_idx,)
#
#
#from dataloaders.video_classification_dataset import VideoClassificationDataset
#from dataloaders.video_sparsesample_dataset import VideoSparsesampleDataset
#from dataloaders.frames_sparsesample_dataset import FramesSparsesampleDataset
#def _get_torch_dataset_split(split, dataloader_type = 'video_clip'):
#
#    mode = split2mode[split]
#    if dataloader_type == 'video_clip':
#        csv_path = os.path.join(video_split_file_dir, split_file_basename[split])
#    elif dataloader_type == 'sparse_video':
#        csv_path = os.path.join(video_split_file_dir, split_file_basename[split])
#    elif dataloader_type == 'sparse_frames':
#        csv_path = os.path.join(frames_split_file_dir, split_file_basename[split])
#    else:
#        raise ValueError('Wrong dataloader type {:s}'.format(dataloader_type))
#
#    return get_torch_dataset(csv_path, mode, dataloader_type=dataloader_type)
#
#
#def _get_unpack_func(dataloader_type = 'video_clip'):
#    if dataloader_type == 'video_clip':
#        return _unpack_data_videoclip
#    elif dataloader_type == 'sparse_video':
#        return _unpack_data_sparse
#    elif dataloader_type == 'sparse_frames':
#        return _unpack_data_sparse
#    else:
#        raise ValueError('Wrong dataloader type {:s}'.format(dataloader_type))
#
#
#def get_torch_dataset(csv_path, mode, dataloader_type = 'video_clip'):
#
#    if dataloader_type == 'video_clip':
#        return VideoClassificationDataset(csv_path, mode,
#            input_frame_length, input_frame_stride, target_fps=input_target_fps, enable_multi_thread_decode=False, decoding_backend='pyav')
#    elif dataloader_type == 'sparse_video':
#        return VideoSparsesampleDataset(csv_path, mode,
#            input_frame_length, target_fps=input_target_fps, enable_multi_thread_decode=False, decoding_backend='pyav')
#    elif dataloader_type == 'sparse_frames':
#        return FramesSparsesampleDataset(csv_path, mode,
#            input_frame_length,)
#    else:
#        raise ValueError('Wrong dataloader type {:s}'.format(dataloader_type))
#
#
#def get_torch_datasets(splits=['train', 'val', 'multicropval'], dataloader_type = 'video_clip'):
#
#    if type(splits) == list:
#        torch_datasets = {}
#        for split in splits:
#            dataset = _get_torch_dataset_split(split, dataloader_type)
#            torch_datasets[split] = dataset
#
#        return torch_datasets, _get_unpack_func(dataloader_type)
#
#    elif type(splits) == str:
#        return _get_torch_dataset_split(splits, dataloader_type), _get_unpack_func(dataloader_type)    
#
#    else:
#        raise ValueError('Wrong type of splits argument. Must be list or string')


'''#{{{
def get_torch_datasets(splits=['train', 'val'], include_coordinates=True):
    from dataloaders.epic_object_feature_dataset_frames import EPIC_object_feature_dataset

    if type(splits) == list:
        torch_datasets = []

        for split in splits:
            dataset = EPIC_object_feature_dataset(train_action_labels_pkl, split_file[split], detection_frame_root, include_coordinates = include_coordinates, return_noun_labels = False)
            torch_datasets.append(dataset)

        return torch_datasets

    elif type(splits) == str:
        return EPIC_object_feature_dataset(train_action_labels_pkl, split_file[splits], detection_frame_root, include_coordinates = include_coordinates, return_noun_labels = False)
    
    else:
        raise ValueError('Wrong type of splits argument. Must be list or string')
'''#}}}


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
