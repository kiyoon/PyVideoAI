"""
Wray multi-verb version of BEOID.
"""
import os
import numpy as np

from video_datasets_api.wray_multiverb.beoid import NUM_CLASSES as num_classes
from video_datasets_api.wray_multiverb.beoid import read_all_annotations
from video_datasets_api.wray_multiverb.beoid import Wray_verb_class_keys_filtered as class_keys 

from pyvideoai.config import DATA_DIR

from pyvideoai.tasks import SingleLabelClassificationTask
task = SingleLabelClassificationTask()


# Paths
dataset_root = os.path.join(DATA_DIR, 'BEOID')
gulp_rgb_dir = os.path.join(dataset_root, 'gulp_rgb')
wray_annotations_root = os.path.join(DATA_DIR, 'Multi-Verb-Labels')
beoid_annotations_root = os.path.join(dataset_root, 'Annotations')

BEOID_all_videos = read_all_annotations(wray_annotations_root, beoid_annotations_root)
video_id_to_multiverb: dict[int, np.array] = {}
for video_info in BEOID_all_videos:
    label_array = np.zeros(num_classes, dtype=float)
    for verb_id in video_info.wray_multiverb_idx:
        label_array[verb_id] = 1.
    video_id_to_multiverb[video_info.clip_id] = label_array

gulp_rgb_split_file_dir = os.path.join(dataset_root, 'wray_splits_gulp_rgb')
split_file_basename_format = {'train': 'train{}.csv', 'val': 'val{}.csv', 'multicropval': 'val{}.csv', 'traindata_testmode': 'train{}.csv'}
split2mode = {'train': 'train', 'val': 'test', 'multicropval': 'test', 'test': 'test', 'traindata_testmode': 'test'}
horizontal_flip = True 

