#!/usr/bin/env python3

import os
import pickle

from video_datasets_api.cater.definitions import NUM_CLASSES_TASK2
from video_datasets_api.cater.cut_scenes import generate_task2_dataset_cut_two_time_slots
from video_datasets_api.cater.generate_labels_from_scenes import generate_task2_labels_from_all_actions, class_keys_task2, class_keys_to_labels

from video_datasets_api.cater.trainval_splits.max2action.actions_order_uniq.train import video_ids as train_video_ids
from video_datasets_api.cater.trainval_splits.max2action.actions_order_uniq.val import video_ids as val_video_ids
from video_datasets_api.cater.trainval_splits.max2action_cameramotion.actions_order_uniq.train import video_ids as cammotion_train_video_ids
from video_datasets_api.cater.trainval_splits.max2action_cameramotion.actions_order_uniq.val import video_ids as cammotion_val_video_ids

import argparse
def get_parser():
    parser = argparse.ArgumentParser(description="Generate CATER task 2 splits, but cut the videos to have two slots only.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("cater_dir", help="cater dir which should consist of max2action and max2action_cameramotion")

    return parser

parser = get_parser()
args = parser.parse_args()


if __name__ == '__main__':

    output_dir = os.path.join(args.cater_dir, "splits_frames_twoslot")
    
    video_ids = {}
    video_ids['max2action'] = {'train': train_video_ids, 'val': val_video_ids}
    video_ids['max2action_cameramotion'] = {'train': cammotion_train_video_ids, 'val': cammotion_val_video_ids}

    CATER_dir = args.cater_dir
    output_CATER_dir = args.cater_dir

    CATER_modes = ['max2action', 'max2action_cameramotion']
    modes = ['containment_only', 'never_containment', 'all']

    classes = class_keys_task2(string=False)
    class_keys_to_labels_dict = class_keys_to_labels(classes)

    for cater_mode in CATER_modes:
        scenes_dir = os.path.join(CATER_dir, cater_mode, 'scenes')
        for mode in modes:
            output_dir = os.path.join(output_CATER_dir, cater_mode, 'twoslot_dataset_splits', 'actions_order_uniq', mode)
            os.makedirs(output_dir, exist_ok=True)

            new_video_id = 0
            for trainval_mode in ['train', 'val']:
                cut_dataset = generate_task2_dataset_cut_two_time_slots(scenes_dir, video_ids[cater_mode][trainval_mode], mode)
                with open(os.path.join(output_dir, f'{trainval_mode}.pkl'), 'wb') as f:
                    pickle.dump(cut_dataset, f, pickle.HIGHEST_PROTOCOL)
                with open(os.path.join(output_dir, f'{trainval_mode}.txt'), 'w') as f:
                    f.write(f'{NUM_CLASSES_TASK2}\n')
                    for cut_dataset_el in cut_dataset:
                        video_id = cut_dataset_el['video_id']
                        start = cut_dataset_el['start']
                        end = cut_dataset_el['end']
                        labels, actions_in_charge = generate_task2_labels_from_all_actions(cut_dataset_el['actions'], class_keys_to_labels_dict)
                        labels_str = ','.join(str(x) for x in sorted(labels))
                        line = f'CATER_new_{video_id:06d}.avi/{{:05d}}.jpg {new_video_id} {labels_str} {start} {end}\n'
                        f.write(line)
                        new_video_id += 1
                        



