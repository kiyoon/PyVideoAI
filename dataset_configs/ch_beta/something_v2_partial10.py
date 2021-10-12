"""
Use only 10% of the data
"""

_exec_relative_('../something_v2.py')

video_split_file_dir = os.path.join(dataset_root, "splits_video_partial10")
frames_split_file_dir = os.path.join(dataset_root, "splits_frames_partial10")
split_file_basename = {'train': 'train_partial.csv', 'val': 'val_partial.csv', 'multicropval': 'val_partial.csv'}
