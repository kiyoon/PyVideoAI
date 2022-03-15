#!/usr/bin/env python3

import tqdm
import pickle
import os
from video_datasets_api.epic_kitchens_100.read_annotations import epic_narration_id_to_unique_id
from gulpio2 import GulpDirectory


import argparse
def get_parser():
    parser = argparse.ArgumentParser(description="Generate EPIC-Kitchens-100 verb-only flow splits. You MUST run epic_convert_rgb_to_flow_frame_idxs.py before running this.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("output_dir", help="Directory to save train.csv and val.csv")
    parser.add_argument("annotations_root_dir", type=str, help="Path to annotations root dir.")
    parser.add_argument("--root", help="Path to the directory with flow files.")
    parser.add_argument("--mode", type=str, default='gulp', choices=['gulp', 'frames'], help="Dataset stored as gulp or extracted frames?")
    parser.add_argument("--verify_num_frames", action='store_true', help="Verify if the number of frames is different from the actual extracted frames and the official annotation.")
    parser.add_argument("--video_ids_pickle", type=str, help="Path to pickle of video IDs to subsample.")

    return parser

parser = get_parser()
args = parser.parse_args()

if args.mode != 'gulp':
    if args.root is None:
        parser.error('For video and frames mode, --root is a required argument.')
else:
    if args.verify_num_frames:
        if args.root is None:
            parser.error('In order to verify the number of frames, --root is a required argument.')

if __name__ == '__main__':
    os.makedirs(args.output_dir, exist_ok=True)
    narration_id_to_uid, _ = epic_narration_id_to_unique_id(args.annotations_root_dir)

    if args.video_ids_pickle is not None:
        with open(args.video_ids_pickle, 'rb') as f:
            video_ids_to_include = pickle.load(f)

    if args.mode == 'gulp' and args.verify_num_frames:
        gulp_dir = GulpDirectory(args.root)

    for split in ['train', 'validation']:
        label_path = os.path.join(args.annotations_root_dir, f'EPIC_100_{split}_flow.pkl')
        if not os.path.isfile(label_path):
            raise FileNotFoundError(f'{label_path} not available. Did you forget to run epic_convert_rgb_to_flow_frame_idxs.py?')
            
        with open(label_path, 'rb') as f:
            epic_action_labels = pickle.load(f)

        # output name is train.csv and val.csv
        if split == 'validation':
            split = 'val'
        
        if args.video_ids_pickle is None:
            split_file = open(os.path.join(args.output_dir, f'{split}.csv'), 'w')
        else:
            split_file = open(os.path.join(args.output_dir, f'{split}_partial.csv'), 'w')

        split_file.write('0')

        num_videos = len(epic_action_labels.index)
        for index in tqdm.tqdm(range(num_videos), desc="Generating splits"):
            narration_id = epic_action_labels.index[index]
            uid = narration_id_to_uid[narration_id]
            if args.video_ids_pickle is not None:
                if uid not in video_ids_to_include:
                    continue

            participant_id = epic_action_labels.participant_id.iloc[index]
            epicvideo_id = epic_action_labels.video_id.iloc[index]
            verb_label = epic_action_labels.verb_class.iloc[index]

            
            start_frame = epic_action_labels.start_frame.iloc[index]
            stop_frame = epic_action_labels.stop_frame.iloc[index]

            if args.mode == 'frames':
                dir_path = os.path.join(args.root, narration_id)
                for i in range(start_frame, stop_frame+1):
                    u_path = os.path.join(args.root, participant_id, epicvideo_id, 'u', f'frame_{i:010d}.jpg')
                    v_path = os.path.join(args.root, participant_id, epicvideo_id, 'v', f'frame_{i:010d}.jpg')
                    assert os.path.isfile(u_path), f'File does not exist: {u_path}'
                    assert os.path.isfile(v_path), f'File does not exist: {v_path}'

                write_str = f'\n{participant_id}/{epicvideo_id}/{{flow_direction}}/frame_{{frame:010d}}.jpg {uid} {verb_label} {start_frame} {stop_frame}'
            elif args.mode == 'gulp':
                # Note that in official EPIC-Kitchens gulp adapter, "stop_frame" is not inclusive.
                # Use gulp adapter from https://github.com/kiyoon/video_datasets_api
                num_frames_annotated = stop_frame - start_frame + 1
                if args.verify_num_frames:
                    num_frames = len(gulp_dir[narration_id])
                    if num_frames != num_frames_annotated:
                        print("narration_id: {:d}, num_frames: {:d}, num_frames_annotated: {:d}".format(narration_id, num_frames, num_frames_annotated))
                write_str = f'\n{narration_id} {uid} {verb_label} 0 {num_frames_annotated-1}'
            else:
                raise NotImplementedError(args.mode)


            split_file.write(write_str)

        split_file.close()
