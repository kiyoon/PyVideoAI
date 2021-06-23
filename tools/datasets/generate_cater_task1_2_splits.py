#!/usr/bin/env python3

import re
import os
import random

from video_datasets_api.cater.definitions import NUM_CLASSES_TASK1, NUM_CLASSES_TASK2

import argparse
def get_parser():
    parser = argparse.ArgumentParser(description="Generate CATER task 1 and 2 splits.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("cater_dir", help="max2action dir")
    parser.add_argument("--partial", action='store_true', help="Saves only part of the videos")
    parser.add_argument("--prob", type=float, default=0.25, help="Probability to include an example. Applied when --partial is specified.")
    parser.add_argument("--seed", type=int, default=12, help="Random seed for partial sampling.")

    return parser

parser = get_parser()
args = parser.parse_args()


if __name__ == '__main__':
    random.seed(args.seed)

    lists_dir = os.path.join(args.cater_dir, "lists")
    if args.partial:
        output_dir = os.path.join(args.cater_dir, "splits_frames_partial")
    else:
        output_dir = os.path.join(args.cater_dir, "splits_frames")
    frames_dir = os.path.join(args.cater_dir, "frames_q5")
    
    action_types = ['actions_order_uniq', 'actions_present']
    num_classes_of_action_types = [NUM_CLASSES_TASK2, NUM_CLASSES_TASK1]
    split_files = ['train.txt', 'val.txt', 'train_subsetT.txt', 'train_subsetV.txt']

    
    def gen_split(input_path, output_path, num_classes):
        with open(input_path, 'r') as input_file:
            with open(output_path, 'w') as output_file:
                output_file.write(f'{num_classes}\n')
                while True:
                    line = input_file.readline()
                    if not line:
                        break

                    if args.partial and random.random() < 1-args.prob:
                        # skip the video
                        continue

                    video_name, labels = line.split(' ')
                    video_name_wo_ext = os.path.splitext(video_name)[0]
                    video_id_str = re.search('CATER_new_(\d+).avi', video_name).group(1)
                    video_id = int(video_id_str)
                    output_line_list = [f'{video_name_wo_ext}/{{:05d}}.jpg', str(video_id), labels.strip(), "0", "300"]

                    # check corrupted video
                    if os.path.isfile(os.path.join(frames_dir, video_name_wo_ext, '00300.jpg')):
                        output_file.write(' '.join(output_line_list) + '\n')
                    else:
                        print(f"Skipping corrupted video: {video_name_wo_ext}")


    for action_type, num_classes in zip(action_types, num_classes_of_action_types):
        os.makedirs(os.path.join(output_dir, action_type), exist_ok=True)
        for split_file in split_files:
            input_path = os.path.join(lists_dir, action_type, split_file)
            output_path = os.path.join(output_dir, action_type, split_file)
            gen_split(input_path, output_path, num_classes)


