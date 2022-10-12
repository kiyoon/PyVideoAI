# https://github.com/mwray/Multi-Verb-Labels
# Read the annotations and see what they look like.

import os
from pprint import pprint
import multiverb_label_inspect


noun_synonyms = {
                'cup': ['mug'],
                'plug': ['socket'],
           }

# some cannot be auto detected so we assign where they belongs to
hard_fix_wray_to_BEOID = {
        'press_button': 'push_button',
        'pull-out_weight-pin': 'pull_weight-pin',
        'scoop_spoon': 'spoon_',
        'insert_foot': 'insert_pedal',
        }



if __name__ == '__main__':
    labels_dir = '/home/s1884147/scratch2/datasets/BEOID/Annotations'

    label_files = os.listdir(labels_dir)

    BEOID_all_labels = set()
    for label_file in label_files:
        with open(os.path.join(labels_dir, label_file), 'r') as f:
            lines = f.readlines()

        for line in lines:
            fields = line.split(',')
            verb_label = fields[2].split('.v.')[0].replace(' ', '-')
            noun_label = fields[3:]
            noun_label = [noun.strip().replace(' ', '-') for noun in noun_label]
            if noun_label[-1] == '':
                del noun_label[-1]
            noun_label = '+'.join(noun_label)
            #print(f"{noun_label = }")

            BEOID_all_labels.add((verb_label, noun_label))

    print("BEOID all labels")
    print(BEOID_all_labels)
    print()

    BEOID_not_used_labels = BEOID_all_labels.copy()

    orig_label_to_multiverb = multiverb_label_inspect.orig_label_to_multiverb('/home/s1884147/scratch2/datasets/GTEA_Gaze_Plus/Multi-Verb-Labels', 'BEOID')

    wray_class_key_to_BEOID_class_keys = {}
    for avail_label in orig_label_to_multiverb.keys():
        avail_verb, avail_noun = avail_label.split('_')
        avail_nouns = avail_noun.split('+')
        assert len(avail_nouns) in [1,2]
        if len(avail_nouns) == 2:
            avail_nouns = [avail_nouns[1], avail_nouns[0]]

        candidates = set()

        for BEOID_label in BEOID_all_labels:
            verb_label, noun_label = BEOID_label

            if avail_verb == verb_label.replace(' ', '-'):
                for avail_noun in avail_nouns:
                    if avail_noun in noun_label.replace(' ', '-'):
                        candidates.add(f'{verb_label}_{noun_label}')
                        BEOID_not_used_labels.discard((verb_label, noun_label))
                        #noun = orig_label.
                    elif avail_noun in noun_synonyms.keys():
                        for synonym in noun_synonyms[avail_noun]:
                            if synonym in noun_label.replace(' ', '-'):
                                candidates.add(f'{verb_label}_{noun_label}')
                                BEOID_not_used_labels.discard((verb_label, noun_label))



        wray_class_key_to_BEOID_class_keys[avail_label] = list(candidates)

    for wray_key, BEOID_key in hard_fix_wray_to_BEOID.items():
        wray_class_key_to_BEOID_class_keys[wray_key].append(BEOID_key)
        verb_label, noun_label = BEOID_key.split('_')
        BEOID_not_used_labels.remove((verb_label, noun_label))



    print("Wray label to BEOID label candidates")
    pprint(wray_class_key_to_BEOID_class_keys)

    BEOID_class_key_to_wray_class_key = {}
    for k, v in wray_class_key_to_BEOID_class_keys.items():
        for beoid_class_key in v:
            BEOID_class_key_to_wray_class_key[beoid_class_key] = k

    print()
    print("BEOID class key to Wray class key")
    pprint(BEOID_class_key_to_wray_class_key)

    print()
    print("BEOID class keys that are not used. No match, and possibly minor classes.")
    pprint(BEOID_not_used_labels)

