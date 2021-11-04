#!/usr/bin/env python3

import tqdm
import pickle
import os
from decord import VideoReader

import argparse
def get_parser():
    parser = argparse.ArgumentParser(description="Generate EPIC-Kitchens splits (P25~P31 are for validation).",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("root", help="Path to the directory with video files.")
    parser.add_argument("output_dir", help="Directory to save train.csv and val.csv")
    parser.add_argument("train_action_labels_pkl", type=str, help="Path to EPIC_train_action_labels.pkl.")
    parser.add_argument("--mode", type=str, default='video', choices=['video', 'frames'], help="Dataset stored as videos or extracted frames?")
    parser.add_argument("--verify_num_frames", action='store_true', help="Verify if the number of frames is different from the actual extracted frames and the official annotation.")
    return parser

parser = get_parser()
args = parser.parse_args()

TRAINING_DATA = ['P' + x.zfill(2) for x in map(str, range(1, 26))]  # P01 to P25

if __name__ == '__main__':
    with open(args.train_action_labels_pkl, 'rb') as f:
        epic_action_labels = pickle.load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    
    train_split = open(os.path.join(args.output_dir, 'train.csv'), 'w')
    val_split = open(os.path.join(args.output_dir, 'val.csv'), 'w')

    train_split.write('0')
    val_split.write('0')

    num_videos = len(epic_action_labels.index)
    for index in tqdm.tqdm(range(num_videos), desc="Generating splits"):
        uid = epic_action_labels.index[index]
        verb_label = epic_action_labels.verb_class.iloc[index]
        #noun_label = epic_action_labels.noun_class.iloc[index]
        participant_id = epic_action_labels.participant_id.iloc[index]

        if args.mode == 'video':
            vr = VideoReader(os.path.join(args.root, f'{uid:05d}.mp4'), num_threads=1)
            num_frames = len(vr)
            height, width, _ = vr[0].shape
            write_str = f'\n{uid:05d}.mp4 {uid} {verb_label} 0 {num_frames-1} {width} {height}'
        elif args.mode == 'frames':
            dir_path = os.path.join(args.root, '%05d' % uid)
            num_frames = len([name for name in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, name))])
            
            # verification with the EPIC_train_action_labels.pkl
            if args.verify_num_frames:
                start_frame = epic_action_labels.start_frame.iloc[index]
                stop_frame = epic_action_labels.stop_frame.iloc[index]
                num_frames_annotated = stop_frame - start_frame + 1

                if num_frames != num_frames_annotated:
                    print("uid: {:d}, num_frames: {:d}, num_frames_annotated: {:d}".format(uid, num_frames, num_frames_annotated))

            write_str = f'\n{uid:05d} {uid} {verb_label} 0 {num_frames-1}'
        else:
            raise NotImplementedError(args.mode)

        if participant_id in TRAINING_DATA:
            train_split.write(write_str)
        else:
            val_split.write(write_str)

    train_split.close()
    val_split.close()
