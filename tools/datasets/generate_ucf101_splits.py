#!/usr/bin/env python3

import tqdm
import pickle
import os

from video_datasets_api.ucf101.class_keys import get_class_keys_to_indices
from video_datasets_api.ucf101.video_id import get_unique_video_ids_from_videos
from video_datasets_api.ucf101.read_splits import read_train_test_splits

import argparse
def get_parser():
    parser = argparse.ArgumentParser(description="Generate UCF101 splits.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("root", help="Path to the directory with video files (extracted as frames).")
    parser.add_argument("output_dir", help="Directory to save trainlist01.txt, testlist01.txt, and so on")
    parser.add_argument("annotations_root", type=str, help="Path to the action recognition annotation directory.")
    return parser

parser = get_parser()
args = parser.parse_args()


if __name__ == '__main__':
    labels_path = os.path.join(args.annotations_root, 'classInd.txt')
    splits = ['trainlist01.txt', 'trainlist02.txt', 'trainlist03.txt', 'testlist01.txt', 'testlist02.txt', 'testlist03.txt']

    class_keys_to_indices = get_class_keys_to_indices(labels_path)
    _, video_ids = get_unique_video_ids_from_videos(args.root, True)

    os.makedirs(args.output_dir, exist_ok=True)
    
    for split_name in splits:
        split_list = read_train_test_splits(os.path.join(args.annotations_root, split_name), class_keys_to_indices)

        with open(os.path.join(args.output_dir, split_name), "w") as out_file:
            # first line: num_classes if multilabel, else 0
            out_file.write("0\n")   # single label classification is always 0
            for path, label in tqdm.tqdm(split_list, desc='Generating splits'):
                path_wo_ext = os.path.splitext(path)[0]
                video_id = video_ids[path]
                dir_path = os.path.join(args.root, f'{path_wo_ext}')
                num_frames = len([name for name in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, name))])
                write_str = f"{path_wo_ext}/{{:05d}}.jpg {video_id} {label} 0 {num_frames-1}\n" 
                out_file.write(write_str)

