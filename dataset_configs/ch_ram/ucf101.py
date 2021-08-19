import os
import numpy as np
import pandas as pd

from video_datasets_api.ucf101.definitions import NUM_CLASSES as num_classes
from video_datasets_api.ucf101.class_keys import get_class_keys
from pyvideoai.config import DATA_DIR

from pyvideoai.tasks import SingleLabelClassificationTask
task = SingleLabelClassificationTask()

# Paths

dataset_root = os.path.join('/dev/shm/datasets/ucf101')
annotations_root = os.path.join(dataset_root, 'ucfTrainTestlist')
#class_keys = pd.DataFrame(get_class_keys(os.path.join(annotations_root, 'classInd.txt')), columns=['class_keys'])['class_keys']
frames_dir = os.path.join(dataset_root, "frames_q5")
frames_split_file_dir = os.path.join(dataset_root, "splits_frames")
split_file_basename1 = {'train': 'trainlist01.txt', 'val': 'testlist01.txt', 'multicropval': 'testlist01.txt'}
split_file_basename2 = {'train': 'trainlist02.txt', 'val': 'testlist02.txt', 'multicropval': 'testlist02.txt'}
split_file_basename3 = {'train': 'trainlist03.txt', 'val': 'testlist03.txt', 'multicropval': 'testlist03.txt'}
split_file_basename = split_file_basename1
split2mode = {'train': 'train', 'val': 'test', 'multicropval': 'test', 'test': 'test'}


# Training settings
horizontal_flip = True 
