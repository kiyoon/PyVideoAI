_exec_relative_('../neighbour_multilabel.py')

from ..epic100_verb.data import best_num_neighbours_per_class, best_thr_per_class

input_type = 'gulp_flow'
loss_type = 'mask_binary_ce'

num_neighbours = 14
thr = 0.2

num_neighbours_per_class = best_num_neighbours_per_class
thr_per_class = best_thr_per_class
