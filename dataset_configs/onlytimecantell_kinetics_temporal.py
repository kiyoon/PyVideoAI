"""
Kinetics-Temporal-32 set (from Kinetics-400) suggested by Only Time Can Tell (WACV 2021, Sevilla-Lara et al.)
"""
import os

from video_datasets_api.onlytimecantell.definitions import NUM_CLASSES_KINETICS_TEMPORAL as num_classes
from pyvideoai.config import DATA_DIR

from pyvideoai.tasks import SingleLabelClassificationTask
task = SingleLabelClassificationTask()

# Paths
dataset_root = os.path.join(DATA_DIR, 'OnlyTimeCanTell', 'kinetics-400')
frames_dir = os.path.join(dataset_root, "frames_q85")
frames_split_file_dir = os.path.join(dataset_root, "splits_frames")
split_file_basename = {'train': 'train_kinetics_temporal.csv', 'val': 'val_kinetics_temporal.csv', 'multicropval': 'val_kinetics_temporal.csv'}
split2mode = {'train': 'train', 'val': 'test', 'multicropval': 'test', 'test': 'test'}

# Training settings
horizontal_flip = False     # because of sthsth
