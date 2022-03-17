# https://github.com/mwray/Multi-Verb-Labels
# Read the annotations and see what they look like.

import pickle
if __name__ == '__main__':
    annotations_root = '/home/s1884147/scratch2/datasets/GTEA_Gaze_Plus/Multi-Verb-Labels'

    with open(annotations_root + '/class_order.csv', 'r') as f:
        class_order = f.read()
    class_order = class_order[1:][:-2].split('","')

    with open(annotations_root + '/GTEA.pkl', 'rb') as f:
        data = pickle.load(f)

    labels = list(data.keys())
    verb_labels = set(label.split('_')[0].replace('-','_') for label in labels)
    print(verb_labels)

    classes_filter_idx = [idx for idx, class_key in enumerate(class_order) if class_key in verb_labels]
    print(classes_filter_idx)

    class_order_filtered = [class_order[idx] for idx in classes_filter_idx]
    print(class_order_filtered)


