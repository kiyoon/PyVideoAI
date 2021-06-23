#!/usr/bin/env python3

import tqdm
import pickle
import os
import random

import video_datasets_api.something_something_v1 as sthv1

import argparse
def get_parser():
    parser = argparse.ArgumentParser(description="Generate Something-Something-V1 splits.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("output_dir", help="Directory to save train.csv and val.csv")
    parser.add_argument("annotations_root", type=str, help="Path to the annotation directory.")
    parser.add_argument("--root", help="Path to the directory with video files.")
    parser.add_argument("--mode", type=str, default='frames', choices=['video', 'frames'], help="Dataset stored as videos or extracted frames?")
    parser.add_argument("--class_sorted", action='store_true', help="When reading the labels.csv, sort the keys and assign numbers. TRN uses this method for some reason.")
    parser.add_argument("--partial", action='store_true', help="Save only partial data.")
    parser.add_argument("--prob", type=float, default=0.25, help="Probability to include an example. Applied when --partial is specified.")
    parser.add_argument("--seed", type=int, default=12, help="Random seed for partial sampling.")
    parser.add_argument("--num_frames_atleast", type=int, default=0, help="Sample only the videos with at least this number of frames.")

    return parser

parser = get_parser()
args = parser.parse_args()

if args.mode == 'frames':
    assert args.root is not None, '--root parameter missing'
else:
    assert args.num_frames_atleast == 0, '--num_frames_atleast cannot come with video input'


if __name__ == '__main__':
    random.seed(args.seed)
    labels_path = os.path.join(args.annotations_root, 'something-something-v1-labels.csv')
    train_path = os.path.join(args.annotations_root, 'something-something-v1-train.csv')
    val_path = os.path.join(args.annotations_root, 'something-something-v1-validation.csv')

    if args.class_sorted:
        labels = sthv1.labels_str2int_sorted(labels_path)
    else:
        labels = sthv1.labels_str2int(labels_path)

    train_uids, train_labels = sthv1.read_splits(train_path, labels)
    val_uids, val_labels = sthv1.read_splits(val_path, labels)

    os.makedirs(args.output_dir, exist_ok=True)
    

    def gen_split(uids, labels, output_path):
        num_samples = len(uids)
        split = open(output_path, 'w')
        split.write('0')  # num_classes but 0 for single label
        for uid, label in tqdm.tqdm(zip(uids, labels), desc="Generating splits", total=num_samples):
            if not args.partial or random.random() < args.prob:
                if args.mode == 'video':
                    write_str = f'\n{uid}.mp4 {uid} {label}'
                elif args.mode == 'frames':
                    dir_path = os.path.join(args.root, f'{uid}')
                    num_frames = len([name for name in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, name))])
                    if num_frames < args.num_frames_atleast:
                        continue
                    write_str = f'\n{uid}/{{:05d}}.jpg {uid} {label} 1 {num_frames}'
                else:
                    raise NotImplementedError(args.mode)

                split.write(write_str)

        split.close()

    def split_name(base_name = 'train'):
        name = base_name
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

