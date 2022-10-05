#!/usr/bin/env python3

import tqdm
import json
import os
import random
from collections import OrderedDict

import video_datasets_api.onlytimecantell.somethingv1_read_annotations as sthv1
import video_datasets_api.onlytimecantell.definitions as onlytimecantell

import argparse
def get_parser():
    parser = argparse.ArgumentParser(description="Generate Kinetics and Something-Something-V1 Temporal/Static/Random splits from Only Time Can Tell (Laura Sevilla et al.).\nDownload Kinetics and create metadata from github.com/kiyoon/kinetics_downloader.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("kinetics_root", help="Path to the directory with video files.")
    parser.add_argument("kinetics_classes", help="Kinetics 400 classes json (list of class keys)")
    parser.add_argument("kinetics_train", help="Kinetics train split json")
    parser.add_argument("kinetics_val", help="Kinetics val split json")
    parser.add_argument("sth_root", help="Path to the directory with video files.")
    parser.add_argument("sth_annotations_root", type=str, help="Path to the annotation directory.")
    parser.add_argument("output_dir", help="Directory to save train.csv and val.csv")
    parser.add_argument("--split", type=str, default='temporal', choices=['temporal', 'static', 'random'], help="The split from Only Time Can Tell.")
    parser.add_argument("--mode", type=str, default='frames', choices=['video', 'frames'], help="Dataset stored as videos or extracted frames?")
    parser.add_argument("--partial", action='store_true', help="Choose 25%% of the data.")
    return parser

parser = get_parser()
args = parser.parse_args()


if __name__ == '__main__':
    train_output_path = os.path.join(args.output_dir, 'train.csv')
    val_output_path = os.path.join(args.output_dir, 'val.csv')
    os.makedirs(args.output_dir, exist_ok=True)

    if args.split == 'temporal':
        kinetics_class_indices = onlytimecantell.KINETICS_TEMPORAL_CLASSES_INDICES
    elif args.split == 'static':
        kinetics_class_indices = onlytimecantell.KINETICS_STATIC_CLASSES_INDICES
    elif args.split == 'random':
        kinetics_class_indices = onlytimecantell.KINETICS_RANDOM_SETS_INDICES
    else:
        raise ValueError()

    with open(args.kinetics_classes, 'r') as f:
        class_keys_400 = json.load(f)
        class_keys_filtered = [class_keys_400[i] for i in kinetics_class_indices]
        labels = {k:v for k, v in zip(class_keys_filtered, range(len(class_keys_filtered)))}
    with open(args.kinetics_train, 'r') as f:
        train_dict = json.load(f, object_pairs_hook=OrderedDict)
    with open(args.kinetics_val, 'r') as f:
        val_dict = json.load(f, object_pairs_hook=OrderedDict)

    def kinetics_gen_split(uid_start_num, labels, trainval_dict, train_or_valid, output_path):
        split = open(output_path, 'w')
        split.write('0\n')
        uid = uid_start_num
        for uuid, class_key in tqdm.tqdm(trainval_dict.items(), desc="Generating splits", total=len(trainval_dict)):
            class_key = class_key['annotations']['label']
            if class_key in labels.keys():      # only the selected classes (temporal/static/random)
                if not args.partial or random.random() < 0.25:      # choose quarter when --partial is set
                    write_str = ''
                    class_key_path = class_key.replace(' ', '_')
                    label = labels[class_key]
                    if args.mode == 'video':
                        write_str = os.path.join(args.kinetics_root, train_or_valid, class_key_path, f'{uuid}.mp4') + ' %d %d\n' % (uid, label)
                    elif args.mode == 'frames':
                        dir_path = os.path.join(args.kinetics_root, train_or_valid, class_key_path, f'{uuid}')
                        if os.path.isdir(dir_path):     # ignore videos not downloaded
                            num_frames = len([name for name in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, name))])
                            if num_frames > 0:  # ignore videos with 0 frames
                                write_str = dir_path + '/{:05d}.jpg %d %d 1 %d\n' % (uid, label, num_frames)
                    else:
                        raise NotImplementedError(args.mode)

                    split.write(write_str)
                    uid += 1

        split.close()

    kinetics_gen_split(1000000, labels, train_dict, 'train', train_output_path)
    kinetics_gen_split(2000000, labels, val_dict, 'valid', val_output_path)

    # SthSth V1


    labels_path = os.path.join(args.sth_annotations_root, 'something-something-v1-labels.csv')
    train_path = os.path.join(args.sth_annotations_root, 'something-something-v1-train.csv')
    val_path = os.path.join(args.sth_annotations_root, 'something-something-v1-validation.csv')

    if args.split == 'temporal':
        sth_class_indices = onlytimecantell.SOMETHINGV1_TEMPORAL_CLASSES_INDICES
    elif args.split == 'static':
        sth_class_indices = onlytimecantell.SOMETHINGV1_STATIC_CLASSES_INDICES
    elif args.split == 'random':
        sth_class_indices = onlytimecantell.SOMETHINGV1_RANDOM_SETS_INDICES
    else:
        raise ValueError()

    labels = sthv1.labels_str2int(labels_path, sth_class_indices)

    train_uids, train_labels = sthv1.read_splits(train_path, labels)
    val_uids, val_labels = sthv1.read_splits(val_path, labels)



    def sth_gen_split(uids, labels, output_path):
        num_samples = len(uids)
        split = open(output_path, 'a')
        split.write('0\n')
        for uid, label in tqdm.tqdm(zip(uids, labels), desc="Generating splits", total=num_samples):

            if not args.partial or random.random() < 0.25:
                if args.mode == 'video':
                    write_str = os.path.join(args.sth_root, '%d.mp4' % uid) + ' %d %d\n' % (uid, label)
                elif args.mode == 'frames':
                    dir_path = os.path.join(args.sth_root, '%d' % uid)
                    num_frames = len([name for name in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, name))])
                    write_str = dir_path + '/{:05d}.jpg %d %d 1 %d\n' % (uid, label, num_frames)
                else:
                    raise NotImplementedError(args.mode)

                split.write(write_str)

        split.close()

    sth_gen_split(train_uids, train_labels, train_output_path)
    sth_gen_split(val_uids, val_labels, val_output_path)
