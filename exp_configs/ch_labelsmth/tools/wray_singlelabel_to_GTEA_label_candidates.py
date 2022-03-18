# https://github.com/mwray/Multi-Verb-Labels
# Read the annotations and see what they look like.

import pickle
import numpy as np
import os

import multiverb_label_inspect
if __name__ == '__main__':
    GTEA_labels_dir = '/home/s1884147/scratch2/datasets/GTEA_Gaze_Plus/labels/labels_cleaned'

    label_files = os.listdir(GTEA_labels_dir)

    orig_label_to_multiverb = multiverb_label_inspect.orig_label_to_multiverb('/home/s1884147/scratch2/datasets/GTEA_Gaze_Plus/Multi-Verb-Labels', 'GTEA')

    for avail_label in orig_label_to_multiverb.keys():
        avail_verb, avail_noun = avail_label.split('_')
        avail_nouns = avail_noun.split('+')
        assert len(avail_nouns) in [1,2]
        if len(avail_nouns) == 2:
            avail_nouns = [avail_nouns[1], avail_nouns[0]]

        print("Wray label: " + str((avail_verb, avail_nouns)))
        candidates = set()
        for label_file in label_files:
            with open(os.path.join(GTEA_labels_dir, label_file), 'r') as f:
                lines = f.readlines()

            for line in lines:
                cut_till = line.find('>')
                verb_label = line[1:cut_till]
                noun_cut_till = line.find('>', cut_till+1)
                noun_label = line[cut_till+2:noun_cut_till]
                #print(f"{noun_label = }")

                if avail_verb == verb_label.replace(' ', '-'):
                    for avail_noun in avail_nouns:
                        if avail_noun in noun_label:
                            candidates.add(f'{verb_label}-{noun_label}')
                            #noun = orig_label.


        print(f"Candidates from GTEA+: {candidates}")
        print()
