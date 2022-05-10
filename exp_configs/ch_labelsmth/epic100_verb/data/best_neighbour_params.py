from __future__ import annotations

import json
import os
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

def jsonKeys2int(x):
    if isinstance(x, dict):
        return {int(k):v for k,v in x.items()}
    return x


with open(os.path.join(SCRIPT_DIR, 'best_num_neighbours_per_class.json'), 'r') as f:
    best_num_neighbours_per_class: dict[int, int] = json.load(f, object_hook=jsonKeys2int)

with open(os.path.join(SCRIPT_DIR, 'best_thr_per_class.json'), 'r') as f:
    best_thr_per_class: dict[int, int] = json.load(f, object_hook=jsonKeys2int)
