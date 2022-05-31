_exec_relative_('../featuremodel.py')

feature_input_type = 'concat_RGB_flow'
loss_type = 'mask_binary_ce'

add_temporal_overlap_as_pseudo_label = True
