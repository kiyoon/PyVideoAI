#!/usr/bin/env python3

import tqdm
import pickle
import os
import random
import numpy as np

import video_datasets_api.something_something_v2 as sthv2
from video_datasets_api.something_something_v2.definitions import NUM_CLASSES

import argparse
def get_parser():
    parser = argparse.ArgumentParser(description="Generate Something-Something-V2 splits for PyVideoAI framework",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("output_dir", help="Directory to save train.csv and val.csv")
    parser.add_argument("annotations_root", type=str, help="Path to the annotation directory.")
    parser.add_argument("--root", help="Path to the directory with video files.")
    parser.add_argument("--mode", type=str, default='frames', choices=['video', 'frames'], help="Dataset stored as videos or extracted frames?")
    parser.add_argument("--partial", action='store_true', help="Save only partial data.")
    parser.add_argument("--prob", type=float, default=0.25, help="Probability to include an example. Applied when --partial is specified.")
    parser.add_argument("--seed", type=int, default=12, help="Random seed for partial sampling.")
    parser.add_argument("--num_frames_atleast", type=int, default=0, help="Sample only the videos with at least this number of frames.")
    parser.add_argument("--head_and_tail", action='store_true', help="Save only head and tail classes (in training data).")
    parser.add_argument("--head_and_tail_count", type=int, default=5, help="Save N head and N tails (thus N*2 classes)")

    return parser

parser = get_parser()
args = parser.parse_args()

if args.mode == 'frames':
    assert args.root is not None, '--root parameter missing'
else:
    assert args.num_frames_atleast == 0, '--num_frames_atleast cannot come with video input'


if __name__ == '__main__':
    random.seed(args.seed)
    labels_path = os.path.join(args.annotations_root, 'something-something-v2-labels.json')
    train_path = os.path.join(args.annotations_root, 'something-something-v2-train.json')
    val_path = os.path.join(args.annotations_root, 'something-something-v2-validation.json')

    labels = sthv2.class_keys_to_int_label(labels_path)

    train_uids, train_labels = sthv2.read_splits(train_path, labels)
    val_uids, val_labels = sthv2.read_splits(val_path, labels)

    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.head_and_tail:
        train_class_frequency = np.bincount(train_labels, minlength=NUM_CLASSES)
        train_class_frequency_argsort = train_class_frequency.argsort()
        train_tails = train_class_frequency_argsort[:args.head_and_tail_count]
        train_heads = train_class_frequency_argsort[-args.head_and_tail_count:]
        train_tails_and_heads = np.concatenate(train_tails, train_heads)
        old_labels_to_new = {}
        for i, old_label in enumerate(train_tails_and_heads):
            old_labels_to_new[old_label] = i    # assign from 0 to 9

    def gen_split(uids, labels, output_path):
        num_samples = len(uids)
        split = open(output_path, 'w')
        split.write('0')  # num_classes but 0 for single label
        for uid, label in tqdm.tqdm(zip(uids, labels), desc="Generating splits", total=num_samples):
            if not args.head_and_tail or label in train_tails_and_heads:
                if not args.partial or random.random() < args.prob:
                    if args.head_and_tail:
                        label = old_labels_to_new[label]    # change label
                    if args.mode == 'video':
                        write_str = f'\n{uid}.mp4 {uid} {label}'
                    elif args.mode == 'frames':
                        dir_path = os.path.join(args.root, f'{uid}')
                        num_frames = len([name for name in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, name))])
                        if num_frames < args.num_frames_atleast:
                            continue
                        write_str = f'\n{uid}/{{:05d}}.jpg {uid} {label} 0 {num_frames-1}'
                    else:
                        raise NotImplementedError(args.mode)

                    split.write(write_str)

        split.close()

    def split_name(base_name = 'train'):
        name = base_name
        if args.head_and_tail:
            name += '_headtail'
        if args.partial:
            name += '_partial'
        if args.num_frames_atleast > 0:
            name += f'_over{args.num_frames_atleast}fr'
        name += '.csv'
        return name

    train_output_path = os.path.join(args.output_dir, split_name('train'))
    val_output_path = os.path.join(args.output_dir, split_name('val'))

    gen_split(train_uids, train_labels, train_output_path)
    gen_split(val_uids, val_labels, val_output_path)

