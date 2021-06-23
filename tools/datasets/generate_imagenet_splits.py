#!/usr/bin/env python3

import os
from video_datasets_api.imagenet.definitions import NUM_VAL_EXAMPLES, NUM_CLASSES
from video_datasets_api.imagenet.read_annotations import parse_meta_mat

import argparse
def get_parser():
    parser = argparse.ArgumentParser(description="Generate ILSVRC2012 train and val splits.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("meta_mat_path", help="Path to ILSVRC2012_devkit_t12/data/meta.mat")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("-t", "--train_dir", help="Imagenet training data dir")
    parser.add_argument("-v", "--val_ground_truth", help="Validation ground truth label txt path")


    return parser

parser = get_parser()
args = parser.parse_args()

if not bool(args.train_dir) and not bool(args.val_ground_truth):
    parser.error("You have to specify at least one of --train_dir or --val_ground_truth.")

if bool(args.train_dir) + bool(args.meta_mat_path) == 1:
    parser.error("For training split generation, you need to specify both --train_dir and --meta_mat_path.")

if __name__ == '__main__':
    os.makedirs(args.output_dir, exist_ok=True)
    val_idx_to_label, wnid_to_label, _ = parse_meta_mat(args.meta_mat_path)

    # Make val split
    if bool(args.val_ground_truth):
        with open(args.val_ground_truth, 'r') as gt:
            groundtruth = gt.read().splitlines()

        assert len(groundtruth) == NUM_VAL_EXAMPLES 

        with open(os.path.join(args.output_dir, 'val.txt'), 'w') as f:
            f.write('0\n')
            for i, val_index in enumerate(groundtruth):
                label = val_idx_to_label[int(val_index)]
                assert label >= 0 and label < NUM_CLASSES
                f.write(f'ILSVRC2012_val_{i+1:08d}.JPEG {i+1} {label}\n')

    # Make train split
    if bool(args.train_dir):
        image_id = NUM_VAL_EXAMPLES + 1
        with open(os.path.join(args.output_dir, 'train.txt'), 'w') as f:
            f.write('0\n')
            for root, dirs, files in os.walk(args.train_dir):
                dirs.sort()     # iterate directories in a sorted manner
                wordnet_id = os.path.basename(root)
                for filename in sorted(files):
                    if filename.lower().endswith('.jpeg'):
                        label = wnid_to_label[wordnet_id]
                        assert label >= 0 and label < NUM_CLASSES
                        f.write(f'{wordnet_id}/{filename} {image_id} {label}\n')
                        image_id += 1
