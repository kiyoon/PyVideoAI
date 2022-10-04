_exec_relative_('sparsesample_RGB_crop224_8frame_largejit_plateau_10scrop.py')

# This will sample 8 frame in greyscale and copy the same channel over the 3 channels.
sampling_mode = 'GreyOnly'
greyscale=True

def _dataloader_shape_to_model_input_shape(inputs):
    N, C, T, H, W = inputs.shape        # C = 1
    return model_cfg.NCTHW_to_model_input_shape(inputs.repeat(1,3,1,1,1))
