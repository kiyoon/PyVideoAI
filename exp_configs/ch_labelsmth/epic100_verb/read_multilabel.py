import pandas as pd
from pyvideoai.config import DATA_DIR
import os
import numpy as np
import json

def read_multilabel():
    data = pd.read_csv(os.path.join(DATA_DIR, 'EPIC_KITCHENS_100', 'ek100-val-multiple-verbs-halfagree-halfconfident-20220427.csv'))
    video_id_to_label = {}
    for video_id, verb_ids in zip(data.oid, data.labels_to_keep_verb_id):
        verb_ids = json.loads(verb_ids)
        label_array = np.zeros(97, dtype=float)
        for verb_id in verb_ids:
            label_array[verb_id] = 1.
        video_id_to_label[video_id] = label_array

    return video_id_to_label

def get_val_holdout_set(video_id_to_label:dict, video_id_to_multilabel: dict):
    """
    Remove video ids from multilabel set.
    """
    return {k: v for k, v in video_id_to_label.items() if k not in video_id_to_multilabel.keys()}

def get_singlemultilabel(video_id_to_label:dict, video_id_to_multilabel: dict):
    """
    Include original single label to multi-label validation set.
    """
    video_id_to_singlemultilabel = {}
    for video_id, label in video_id_to_multilabel.items():
        singlemultilabel = label.copy()
        singlemultilabel[video_id_to_label[video_id]] = 1.
        video_id_to_singlemultilabel[video_id] = singlemultilabel

    return video_id_to_singlemultilabel
