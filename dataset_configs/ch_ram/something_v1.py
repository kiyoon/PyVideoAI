"""
Use shared memory instead of filesystem.
"""
_exec_relative_('../something_v1.py')

# Paths
dataset_root = '/dev/shm/datasets/something-something-v1'
frames_dir = os.path.join(dataset_root, 'frames')
annotations_root = os.path.join(dataset_root, 'annotations')
label_csv = os.path.join(annotations_root, 'something-something-v1-labels.csv')
split_file = {'train': 'something-something-v1-train.csv', 'val': 'something-something-v1-validation.csv'}
for key in split_file.keys():
    split_file[key] = os.path.join(annotations_root, split_file[key])

video_split_file_dir = os.path.join(dataset_root, "splits_video")
frames_split_file_dir = os.path.join(dataset_root, "splits_frames")
split_file_basename = {'train': 'train.csv', 'val': 'val.csv', 'multicropval': 'val.csv'}
split2mode = {'train': 'train', 'val': 'test', 'multicropval': 'test', 'test': 'test'}
