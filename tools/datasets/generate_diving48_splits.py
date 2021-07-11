#!/usr/bin/env python3

import tqdm
import pickle
import random
import os

import video_datasets_api.diving48 as diving48 

import argparse
def get_parser():
    parser = argparse.ArgumentParser(description="Generate diving48 splits.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("output_dir", help="Directory to save train.csv and val.csv")
    parser.add_argument("annotations_root", type=str, help="Path to the annotation directory.")
    parser.add_argument("--mode", type=str, default='frames', choices=['video', 'frames'], help="Dataset stored as videos or extracted frames?")
    parser.add_argument("--partial", action='store_true', help="Choose 25%% of the data.")
    parser.add_argument("--seed", default=12, help="Partial data random seed")
    parser.add_argument("--V2", action='store_true', help="Use V2 (30/10/2020 update)")
    return parser

parser = get_parser()
args = parser.parse_args()


if __name__ == '__main__':
    random.seed(args.seed)
    if args.V2:
        train_path = os.path.join(args.annotations_root, 'Diving48_V2_train.json')
        test_path = os.path.join(args.annotations_root, 'Diving48_V2_test.json')
    else:
        train_path = os.path.join(args.annotations_root, 'Diving48_train.json')
        test_path = os.path.join(args.annotations_root, 'Diving48_test.json')

    train_vid_names, train_labels, train_num_frames = diving48.read_splits(train_path)
    test_vid_names, test_labels, test_num_frames = diving48.read_splits(test_path)

    os.makedirs(args.output_dir, exist_ok=True)
    

    def gen_split(vid_names, labels, num_frames, output_path):
        num_samples = len(vid_names)
        split = open(output_path, 'w')
        split.write('0\n')  # header: 0 indicates this is single label (no one-hot).
        for uid, (vid_name, label, num_frame) in tqdm.tqdm(enumerate(zip(vid_names, labels, num_frames)), desc="Generating splits", total=num_samples):
            if not args.partial or (args.partial and random.random() < 0.25):
                if args.mode == 'video':
                    write_str = f'{vid_name}.mp4 {uid} {label}\n'
                elif args.mode == 'frames':
                    write_str = f'{vid_name}/{{:05d}}.jpg {uid} {label} 0 {num_frame-1}\n'
                else:
                    raise NotImplementedError(args.mode)

                split.write(write_str)

        split.close()

    def split_name(base_name = 'train'):
        name = base_name
        if args.V2:
            name += '_V2'
        if args.partial:
            name += '_partial'
        name += '.csv'
        return name

    train_output_path = os.path.join(args.output_dir, split_name('train'))
    gen_split(train_vid_names, train_labels, train_num_frames, train_output_path)

    test_output_path = os.path.join(args.output_dir, split_name('test'))
    gen_split(test_vid_names, test_labels, test_num_frames, test_output_path)

