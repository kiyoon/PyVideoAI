_exec_relative_('densesample_RGB_crop224_8x8_largejit_plateau_3scrop10tcrop.py')

sampling_mode = 'TC'

from pyvideoai.utils.tc_reordering import NCTHW_to_TC_NCTHW
def _dataloader_shape_to_model_input_shape(inputs):
    return model_cfg.NCTHW_to_model_input_shape(NCTHW_to_TC_NCTHW(inputs))
