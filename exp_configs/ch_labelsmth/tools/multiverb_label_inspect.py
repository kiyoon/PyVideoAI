# https://github.com/mwray/Multi-Verb-Labels
# Read the annotations and see what they look like.

import pickle
import numpy as np

def orig_label_to_multiverb(annotations_root, dataset):
    with open(annotations_root + '/class_order.csv', 'r') as f:
        class_order = f.read()
    class_order = class_order[1:][:-2].split('","')

    with open(annotations_root + f'/{dataset}.pkl', 'rb') as f:
        data = pickle.load(f)

    labels = list(data.keys())
    verb_labels = set(label.split('_')[0].replace('-','_') for label in labels)
    if dataset == 'CMU':
        verb_labels.add('twist')
        verb_labels.remove('twist_on')
        verb_labels.remove('twist_off')


    classes_filter_idx = [idx for idx, class_key in enumerate(class_order) if class_key in verb_labels]
    #print(classes_filter_idx)

    class_order_filtered = [class_order[idx] for idx in classes_filter_idx]
    assert len(verb_labels) == len(class_order_filtered)

    ret = {}
    for orig_label, multiverb_label in data.items():
        ret[orig_label] = []
        multiverb_label_filtered = [multiverb_label[idx] for idx in classes_filter_idx]
        #print(multiverb_label_filtered)

        sort_indices = np.argsort(multiverb_label_filtered)[::-1]
        for sort_idx in sort_indices:
            if multiverb_label_filtered[sort_idx] >= 0.2:
                ret[orig_label].append(class_order_filtered[sort_idx])

    return ret

if __name__ == '__main__':

    annotations_root = '/home/s1884147/scratch2/datasets/GTEA_Gaze_Plus/Multi-Verb-Labels'

    with open(annotations_root + '/class_order.csv', 'r') as f:
        class_order = f.read()
    class_order = class_order[1:][:-2].split('","')

    print('new 90 labels')
    print(class_order)

    for dataset in ['GTEA', 'CMU', 'BEOID']:
        with open(annotations_root + f'/{dataset}.pkl', 'rb') as f:
            data = pickle.load(f)

        labels = list(data.keys())
        verb_labels = set(label.split('_')[0].replace('-','_') for label in labels)
        if dataset == 'CMU':
            verb_labels.add('twist')
            verb_labels.remove('twist_on')
            verb_labels.remove('twist_off')
        print()
        print(f"{dataset} All labels")
        print(labels)
        print()
        print("Verb labels")
        print(verb_labels)
        print()


        classes_filter_idx = [idx for idx, class_key in enumerate(class_order) if class_key in verb_labels]
        #print(classes_filter_idx)

        class_order_filtered = [class_order[idx] for idx in classes_filter_idx]
        print("Verb labels from new 90 labels")
        print(class_order_filtered)
        print()
        assert len(verb_labels) == len(class_order_filtered)

        for orig_label, multiverb_label in data.items():
            print(f"{orig_label = }")
            multiverb_label_filtered = [multiverb_label[idx] for idx in classes_filter_idx]
            
            # l1 norm
            sum_filtered = sum(multiverb_label_filtered)
            multiverb_label_filtered = [score / sum_filtered for score in multiverb_label_filtered]
            #print(multiverb_label_filtered)

            sort_indices = np.argsort(multiverb_label_filtered)[::-1]
            for sort_idx in sort_indices:
                if multiverb_label_filtered[sort_idx] >= 0.15:
                    print(class_order_filtered[sort_idx], round(multiverb_label_filtered[sort_idx], 2))

