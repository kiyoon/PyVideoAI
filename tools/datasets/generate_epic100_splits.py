#!/usr/bin/env python3

import tqdm
import pickle
import os
from decord import VideoReader
from video_datasets_api.epic_kitchens_100.read_annotations import epic_narration_id_to_unique_id


import argparse
def get_parser():
    parser = argparse.ArgumentParser(description="Generate EPIC-Kitchens-100 verb-only splits",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("root", help="Path to the directory with video files.")
    parser.add_argument("output_dir", help="Directory to save train.csv and val.csv")
    parser.add_argument("annotations_root_dir", type=str, help="Path to annotations root dir.")
    parser.add_argument("--mode", type=str, default='video', choices=['video', 'frames'], help="Dataset stored as videos or extracted frames?")
    parser.add_argument("--verify_num_frames", action='store_true', help="Verify if the number of frames is different from the actual extracted frames and the official annotation.")
    return parser

parser = get_parser()
args = parser.parse_args()

if __name__ == '__main__':
    os.makedirs(args.output_dir, exist_ok=True)
    narration_id_to_uid, _ = epic_narration_id_to_unique_id(args.annotations_root_dir)

    for split in ['train', 'validation']:
        with open(os.path.join(args.annotations_root_dir, f'EPIC_100_{split}.pkl'), 'rb') as f:
            epic_action_labels = pickle.load(f)

        # output name is train.csv and val.csv
        if split == 'validation':
            split = 'val'
        
        split_file = open(os.path.join(args.output_dir, f'{split}.csv'), 'w')

        split_file.write('0')

        num_videos = len(epic_action_labels.index)
        for index in tqdm.tqdm(range(num_videos), desc="Generating splits"):
            narration_id = epic_action_labels.index[index]
            uid = narration_id_to_uid[narration_id]
            verb_label = epic_action_labels.verb_class.iloc[index]

            if args.mode == 'video':
                vr = VideoReader(os.path.join(args.root, f'{uid:05d}.mp4'), num_threads=1)
                num_frames = len(vr)
                height, width, _ = vr[0].shape
                write_str = f'\n{narration_id}.mp4 {uid} {verb_label} 0 {num_frames-1} {width} {height}'
            elif args.mode == 'frames':
                dir_path = os.path.join(args.root, narration_id)
                num_frames = len([name for name in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, name))])
                
                # verification with the EPIC_train_action_labels.pkl
                if args.verify_num_frames:
                    start_frame = epic_action_labels.start_frame.iloc[index]
                    stop_frame = epic_action_labels.stop_frame.iloc[index]
                    num_frames_annotated = stop_frame - start_frame + 1

                    if num_frames != num_frames_annotated:
                        print("uid: {:d}, num_frames: {:d}, num_frames_annotated: {:d}".format(uid, num_frames, num_frames_annotated))

                write_str = f'\n{narration_id} {uid} {verb_label} 0 {num_frames-1}'
            else:
                raise NotImplementedError(args.mode)

        split_file.close()
