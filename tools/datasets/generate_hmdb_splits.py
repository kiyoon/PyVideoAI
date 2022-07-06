#!/usr/bin/env python3

import os
import random
from functools import lru_cache
import tqdm
from video_datasets_api.hmdb.read_annotations import get_unique_video_id, get_class_keys
from video_datasets_api.hmdb.definitions import NUM_CLASSES
from gulpio2 import GulpDirectory



import argparse
def get_parser():
    parser = argparse.ArgumentParser(description="Generate HMDB splits",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("video_dir", help="Path to the directory of video (Gulp) files")
    parser.add_argument("output_dir", help="Directory to save train.csv and test.csv")
    parser.add_argument("annotations_root_dir", type=str, help="Path to annotations root dir.")
    parser.add_argument("--mode", type=str, default='gulp', choices=['gulp', 'video', 'frames'], help="Dataset stored as gulp, videos or extracted frames?")
    parser.add_argument("--confusion", type=int, default=1, help="Add confusion to the dataset. If the value is 2 or bigger, it will assign more than 51 classes (102 classes for 2) and randomly assign train label. Test labels will be multi-label.")
    parser.add_argument("--seed", type=int, default=12, help="Random seed for the confusion.")
    return parser

parser = get_parser()
args = parser.parse_args()

if args.mode != 'gulp':
    parser.error('Only supports gulp mode for now.')

if args.confusion < 1:
    parser.error('--confusion needs to be 1 or higher.')

if __name__ == '__main__':
    random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    filename_to_video_id, _ = get_unique_video_id()
    classes = get_class_keys()

    if args.mode == 'gulp':
        gulp_dir = GulpDirectory(args.video_dir)

        # Reading metadata from gulp is slow.. Cache the num_frames.
        @lru_cache(maxsize=None)
        def get_num_frames(video_name):
            return gulp_dir[video_name][1]['num_frames']

    for split_num in range(1, 4):
        print(f'{split_num = }')
        with open(os.path.join(args.output_dir, f'train{split_num}.csv'), 'w') as train_file:
            train_file.write('0\n')
            with open(os.path.join(args.output_dir, f'test{split_num}.csv'), 'w') as test_file:
                if args.confusion == 1:
                    test_file.write('0\n')
                else:
                    test_file.write(f'{NUM_CLASSES * args.confusion}\n')

                for class_idx, class_name in enumerate(classes):
                    print(class_name)
                    train_count = 0
                    test_count = 0
                    with open(os.path.join(args.annotations_root_dir, f'{class_name}_test_split{split_num}.txt'), 'r') as f:
                        for line in tqdm.tqdm(f.read().splitlines()):
                            filename, split = line.split()
                            filename = os.path.splitext(filename)[0]
                            filename = f'{class_name}/{filename}'
                            video_id = filename_to_video_id[filename]
                            num_frames = get_num_frames(filename)
                            if split == '1':
                                # train
                                confusion_idx = random.randint(0, args.confusion - 1)
                                label = confusion_idx * NUM_CLASSES + class_idx
                                train_file.write(f'{filename} {video_id} {label} 0 {num_frames-1}\n')
                                train_count += 1
                            elif split == '2':
                                # test
                                labels = [str(i * NUM_CLASSES + class_idx) for i in range(args.confusion)]
                                labels_str = ','.join(labels)
                                test_file.write(f'{filename} {video_id} {labels_str} 0 {num_frames-1}\n')
                                test_count += 1

                    assert train_count == 70, f'{train_count}'
                    assert test_count == 30
