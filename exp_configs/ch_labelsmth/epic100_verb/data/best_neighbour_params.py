from __future__ import annotations

import json
import os
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(SCRIPT_DIR, 'best_num_neighbours_per_class.json'), 'r') as f:
    best_num_neighbours_per_class: dict[int, int] = json.load(f)

with open(os.path.join(SCRIPT_DIR, 'best_thr_per_class.json'), 'r') as f:
    best_thr_per_class: dict[int, int] = json.load(f)
