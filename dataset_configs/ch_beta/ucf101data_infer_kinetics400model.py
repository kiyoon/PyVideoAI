# Use Kinetics400 pretrained model, but infer using UCF101 data.

import os
import numpy as np
import pandas as pd

from video_datasets_api.kinetics400 import NUM_CLASSES as num_classes
from video_datasets_api.kinetics400 import CLASS_KEYS as kinetics400_class_keys 
from video_datasets_api.ucf101.class_keys import get_class_keys
from pyvideoai.config import DATA_DIR

from pyvideoai.tasks import SingleLabelClassificationTask
task = SingleLabelClassificationTask()

# Paths
dataset_root = os.path.join(DATA_DIR, 'ucf101')
annotations_root = os.path.join(dataset_root, 'UCF101_Action_recognition_splits')
class_key_txt = os.path.join(annotations_root, 'classInd.txt')
ucf101_class_keys = get_class_keys(class_key_txt)

frames_dir = os.path.join(dataset_root, "frames")
frames_split_file_dir = os.path.join(dataset_root, "splits_frames")
split_file_basename1 = {'train': 'trainlist01.txt', 'val': 'testlist01.txt', 'multicropval': 'testlist01.txt'}
split_file_basename2 = {'train': 'trainlist02.txt', 'val': 'testlist02.txt', 'multicropval': 'testlist02.txt'}
split_file_basename3 = {'train': 'trainlist03.txt', 'val': 'testlist03.txt', 'multicropval': 'testlist03.txt'}
split2mode = {'train': 'train', 'val': 'test', 'multicropval': 'test', 'test': 'test'}
