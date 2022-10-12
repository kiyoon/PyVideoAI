"""
Get rid of temporal overlapping segments in train set.
"""

import pandas as pd

TRAIN_CSV_PATH = '../../../../../data/EPIC_KITCHENS_100/splits_gulp_flow/train.csv'
OVERLAP_CSV_PATH = '/home/kiyoon/project/multi-label-ar/annotations/ek_100_train_overlapping/EK100_train_overlapping_extension=0.csv'

OUTPUT_CSV_PATH = '../../../../../data/EPIC_KITCHENS_100/splits_gulp_flow/train_discard_overlap_extension=0.csv'


if __name__ == '__main__':
    overlaps = pd.read_csv(OVERLAP_CSV_PATH)

    with open(TRAIN_CSV_PATH, 'r') as input_file:
        with open(OUTPUT_CSV_PATH, 'w') as output_file:
            count = 0
            for idx, line in enumerate(input_file):
                if idx == 0:
                    # Just a header
                    output_file.write(line)
                    continue

                narration_id = line.strip().split(' ')[0]

                if narration_id in (overlaps.a.tolist() + overlaps.b.tolist()):
                    output_file.write(line)
                    count += 1
                    print(count, idx)
