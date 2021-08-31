_exec_relative_('../ucf101.py')
# Paths

dataset_root = os.path.join('/dev/shm/datasets/ucf101')
annotations_root = os.path.join(dataset_root, 'ucfTrainTestlist')
frames_dir = os.path.join(dataset_root, "frames_q5")
frames_split_file_dir = os.path.join(dataset_root, "splits_frames")
