_exec_relative_('sparsesample_RGB_crop224_8frame_largejit_plateau_10scrop.py')

sampling_mode = 'TC'

from pyvideoai.utils.tc_reordering import NCTHW_to_TC_NCTHW, TCrgb_idx
def _dataloader_shape_to_model_input_shape(inputs):
    return model_cfg.NCTHW_to_model_input_shape(NCTHW_to_TC_NCTHW(inputs, TCrgb_idx))
