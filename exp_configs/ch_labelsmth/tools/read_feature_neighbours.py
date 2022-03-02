import argparse
import os
import pickle
import numpy as np
import re

from video_datasets_api.epic_kitchens_100.get_class_keys import EPIC100_get_class_keys

def get_parser():
    parser = argparse.ArgumentParser(description="See how multi pseudo labels change.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('features_neighbours_dir')
    parser.add_argument('--epic100_annotations_root', default = '/home/s1884147/project/PyVideoAI/data/EPIC_KITCHENS_100/epic-kitchens-100-annotations')
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    regex_format = 'feature_neighbours_epoch_(\d+).pkl'
    file_list = os.listdir(args.features_neighbours_dir)

    pickles = sorted([filename for filename in file_list if re.search(regex_format, filename) is not None])
    epochs = [int(re.search(regex_format, filename).group(1)) for filename in pickles]

    print("Reading pickles..")
    print(pickles)

    video_ids = []
    gt_labels = []
    multilabels = []
    for pickle_file in pickles:
        with open(os.path.join(args.features_neighbours_dir, pickle_file), 'rb') as f:
            data = pickle.load(f)
            multilabels.append(data['multilabels'])
            video_ids.append(data['video_ids'])
            gt_labels.append(data['labels'])

    multilabels = np.stack(multilabels)
    video_ids = np.stack(video_ids)
    gt_labels = np.stack(gt_labels)
    
    print(multilabels.shape)

    verb_class_keys = EPIC100_get_class_keys(args.epic100_annotations_root, 'verb')

    for video_idx in range(100, 150):
        print(f'video_id = {video_ids[0,video_idx]}')
        print(f'GT label = {verb_class_keys[gt_labels[0, video_idx]]}')
        for epoch, video_id, gt_label, multilabel in zip(epochs, video_ids, gt_labels, multilabels):
            pseudo_labels = np.concatenate(np.argwhere(multilabel[video_idx] == 1))
            string_labels = [str(verb_class_keys[pseudo_label]) for pseudo_label in pseudo_labels] 
            print(f'{epoch = }, {string_labels}')
    

if __name__ == '__main__':
    main()
