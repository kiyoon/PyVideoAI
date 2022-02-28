from pyvideoai.config import DATA_DIR
import os
import numpy as np

def read_multilabel():
    data = pd.read_csv(os.path.join(DATA_DIR, 'ek100-val-multiple-verbs-minf2.csv'))
    video_id_to_label = {}
    for video_id, verb_ids in zip(data.oid, data.labels_to_keep_verb_id):
        verb_ids = json.loads(verb_ids)
        label_array = np.zeros(97, dtype=float)
        for verb_id in verb_ids:
            label_array[verb_id] = 1.
        video_id_to_label[video_id] = label_array

    return video_id_to_label
