import argparse
import os
import pickle
import numpy as np
import re

from video_datasets_api.epic_kitchens_100.get_class_keys import EPIC100_get_class_keys
from video_datasets_api.epic_kitchens_100.read_annotations import epic_narration_id_to_unique_id

def get_parser():
    parser = argparse.ArgumentParser(description="See how multi pseudo labels change.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('features_neighbours_dir')
    parser.add_argument('--epic100_annotations_root', default = '/home/s1884147/project/PyVideoAI/data/EPIC_KITCHENS_100/epic-kitchens-100-annotations')
    parser.add_argument('--thr', type=float, default = 0.2)
    return parser


            
def get_label_change_info(features_neighbours_dir):
    regex_format = 'feature_neighbours_epoch_(\d+).pkl'
    file_list = os.listdir(features_neighbours_dir)

    pickles = sorted([filename for filename in file_list if re.search(regex_format, filename) is not None])
    epochs = [int(re.search(regex_format, filename).group(1)) for filename in pickles]

    print("Reading pickles..")
    print(pickles)

    video_ids = []
    gt_labels = []
    neighbour_class_frequency = []
    for pickle_file in pickles:
        with open(os.path.join(features_neighbours_dir, pickle_file), 'rb') as f:
            data = pickle.load(f)
            neighbour_class_frequency.append(data['nc_freq'])
            video_ids.append(data['video_ids'])
            gt_labels.append(data['labels'])

    neighbour_class_frequency = np.stack(neighbour_class_frequency)
    video_ids = np.stack(video_ids)
    gt_labels = np.stack(gt_labels)
    
    
    pseudo_soft_labels = neighbour_class_frequency / np.sum(neighbour_class_frequency, axis=-1, keepdims=True)
    
    return epochs, video_ids, gt_labels, pseudo_soft_labels

def main():
    parser = get_parser()
    args = parser.parse_args()
    epochs, video_ids, gt_labels, pseudo_soft_labels = get_label_change_info(args.features_neighbours_dir)
    
    verb_class_keys = EPIC100_get_class_keys(args.epic100_annotations_root, 'verb')
    _, video_id_to_narration_id = epic_narration_id_to_unique_id(args.epic100_annotations_root)  

    for video_idx in range(100, 150):
        video_id = video_id_to_narration_id[video_ids[0,video_idx]]
        print(f'{video_id = }')
        print(f'GT label = {verb_class_keys[gt_labels[0, video_idx]]}')
        for epoch, video_id, gt_label, pseudo_soft_label in zip(epochs, video_ids, gt_labels, pseudo_soft_labels):
            multilabel = (pseudo_soft_label[video_idx] > args.thr).astype(int)
            pseudo_labels = np.concatenate(np.argwhere(multilabel == 1))
            string_labels = [str(verb_class_keys[pseudo_label]) for pseudo_label in pseudo_labels] 
            print(f'{epoch = }, {string_labels}')

if __name__ == '__main__':
    main()
