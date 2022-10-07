import pandas as pd
from pyvideoai.config import DATA_DIR
import os
import numpy as np
import json


def read_multiverb(narration_id_to_video_id):
    """
    Read from the new format
    """
    data = pd.read_csv(os.path.join(DATA_DIR, 'EPIC_KITCHENS_100', 'ek100-val-multiple-verbs-halfagree-halfconfident-include_original-20220427.csv'))
    video_id_to_label = {}
    for narration_id, verb_ids in zip(data.narration_id, data.verb_labels):
        verb_ids = json.loads(verb_ids)
        label_array = np.zeros(97, dtype=float)
        for verb_id in verb_ids:
            label_array[verb_id] = 1.
        video_id = narration_id_to_video_id[narration_id]
        video_id_to_label[video_id] = label_array

    return video_id_to_label


def get_val_holdout_set(video_id_to_label:dict, video_id_to_multilabel: dict):
    """
    Remove video ids from multilabel set.
    """
    return {k: v for k, v in video_id_to_label.items() if k not in video_id_to_multilabel.keys()}
