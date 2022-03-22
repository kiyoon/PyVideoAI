# https://github.com/mwray/Multi-Verb-Labels
# Read the annotations and see what they look like.

import os

import multiverb_label_inspect
if __name__ == '__main__':
    labels_dir = '/home/s1884147/scratch2/datasets/CMU-MMAC/annotations'

    label_dirs = os.listdir(labels_dir)

    CMU_all_labels = set()
    for label_dir in label_dirs:
        with open(os.path.join(labels_dir, label_dir, 'labels.dat'), 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.replace('\n', '')
            fields = line.split(' ')
            label = fields[2].split('-')
            verb_label = label[0]
            noun_label = '-'.join(label[1:])
            #print(f"{noun_label = }")

            CMU_all_labels.add((verb_label, noun_label))

    print("CMU-MMAC all labels")
    print(CMU_all_labels)

    orig_label_to_multiverb = multiverb_label_inspect.orig_label_to_multiverb('/home/s1884147/scratch2/datasets/GTEA_Gaze_Plus/Multi-Verb-Labels', 'CMU')

    for avail_label in orig_label_to_multiverb.keys():
        avail_label = avail_label.split('_')
        avail_verb = avail_label[0]
        avail_noun = avail_label[1] if len(avail_label) > 1 else ''
        avail_nouns = avail_noun.split('+')
        assert len(avail_nouns) in [1,2]
        if len(avail_nouns) == 2:
            avail_nouns = [avail_nouns[1], avail_nouns[0]]

        print("Wray label: " + str((avail_verb, avail_nouns)))
        candidates = set()

        for CMU_label in CMU_all_labels:
            verb_label, noun_label = CMU_label

            if avail_verb == verb_label.replace('_', '-'):
                for avail_noun in avail_nouns:
                    if avail_noun.replace('_', ' ').replace('-', ' ') in noun_label.replace('_', ' ').replace('-', ' '):
                        candidates.add(f'{verb_label}-{noun_label}')
                        #noun = orig_label.


        print(f"Candidates from CMU: {candidates}")
        print()


