_exec_relative_('../something_v2.py')
# Paths
dataset_root = os.path.join(DATA_DIR, 'something-something-v2')
frames_dir = '/dev/shm/datasets/something-something-v2/frames_q5'
annotations_root = os.path.join(dataset_root, 'annotations')
label_json = os.path.join(annotations_root, 'something-something-v2-labels.json')

video_split_file_dir = os.path.join(dataset_root, "splits_video")
frames_split_file_dir = os.path.join(dataset_root, "splits_frames")
split_file_basename = {'train': 'train.csv', 'val': 'val.csv', 'multicropval': 'val.csv'}
split2mode = {'train': 'train', 'val': 'test', 'multicropval': 'test', 'test': 'test'}
