# https://github.com/mwray/Multi-Verb-Labels
# Read the annotations and see what they look like.

import os

import multiverb_label_inspect
if __name__ == '__main__':
    labels_dir = '/home/s1884147/scratch2/datasets/BEOID/Annotations'

    label_files = os.listdir(labels_dir)

    BEOID_all_labels = set()
    for label_file in label_files:
        with open(os.path.join(labels_dir, label_file), 'r') as f:
            lines = f.readlines()

        for line in lines:
            fields = line.split(',')
            verb_label = fields[2].split('.v.')[0]
            noun_label = fields[3:]
            noun_label = [noun.strip() for noun in noun_label]
            if noun_label[-1] == '':
                del noun_label[-1]
            noun_label = ','.join(noun_label)
            #print(f"{noun_label = }")

            BEOID_all_labels.add((verb_label, noun_label))

    print("BEOID all labels")
    print(BEOID_all_labels)

    orig_label_to_multiverb = multiverb_label_inspect.orig_label_to_multiverb('/home/s1884147/scratch2/datasets/GTEA_Gaze_Plus/Multi-Verb-Labels', 'BEOID')

    for avail_label in orig_label_to_multiverb.keys():
        avail_verb, avail_noun = avail_label.split('_')
        avail_nouns = avail_noun.split('+')
        assert len(avail_nouns) in [1,2]
        if len(avail_nouns) == 2:
            avail_nouns = [avail_nouns[1], avail_nouns[0]]

        #print("Wray label: " + str((avail_verb, avail_nouns)))
        print("Wray label: " + avail_label)
        candidates = set()

        for BEOID_label in BEOID_all_labels:
            verb_label, noun_label = BEOID_label

            if avail_verb == verb_label.replace(' ', '-'):
                for avail_noun in avail_nouns:
                    if avail_noun in noun_label.replace(' ', '-'):
                        candidates.add(f'{verb_label}-{noun_label}')
                        #noun = orig_label.


        print(f"Candidates from BEOID: {candidates}")
        print()


