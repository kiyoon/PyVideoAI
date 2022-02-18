#!/usr/bin/env python3

import tqdm
import pickle
import os
from decord import VideoReader
from video_datasets_api.epic_kitchens_100.read_annotations import epic_narration_id_to_unique_id


import argparse
def get_parser():
    parser = argparse.ArgumentParser(description="Generate EPIC-Kitchens-100 verb-only flow splits. You MUST run epic_convert_rgb_to_flow_frame_idxs.py before running this.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("root_55", help="Path to the directory with flow files. (EPIC-55 directory structure)")
    parser.add_argument("root_100", help="Path to the directory with flow files. (EPIC-100 extensions directory)")
    parser.add_argument("output_dir", help="Directory to save train.csv and val.csv")
    parser.add_argument("annotations_root_dir", type=str, help="Path to annotations root dir.")
    parser.add_argument("--video_ids_pickle", type=str, help="Path to pickle of video IDs to subsample.")

    return parser

parser = get_parser()
args = parser.parse_args()

if __name__ == '__main__':
    os.makedirs(args.output_dir, exist_ok=True)
    narration_id_to_uid, _ = epic_narration_id_to_unique_id(args.annotations_root_dir)

    if args.video_ids_pickle is not None:
        with open(args.video_ids_pickle, 'rb') as f:
        video_ids_to_include = pickle.load(f)

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
            verb_label = epic_action_labels.verb_class.iloc[index]

            dir_path = os.path.join(args.root, narration_id)
            
            start_frame = epic_action_labels.start_frame.iloc[index]
            stop_frame = epic_action_labels.stop_frame.iloc[index]

            write_str = f'\n{split}/{narration_id} {uid} {verb_label} {start_frame} {stop_frame}'

            split_file.write(write_str)

        split_file.close()
