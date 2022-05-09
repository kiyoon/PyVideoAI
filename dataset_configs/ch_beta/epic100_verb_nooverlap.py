"""
Discard all segments that has temporal overlap with another (in training data).
Idea is to see if it trains better with more accurate segments.
"""

_exec_relative_('../epic100_verb.py')

split_file_basename['train'] = 'train_discard_overlap_extension=0.csv'
split_file_basename['traindata_testmode'] = 'train_discard_overlap_extension=0.csv'
