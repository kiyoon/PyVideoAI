_exec_relative_('densesample_RGB_crop224_8x8_largejit_plateau_3scrop10tcrop.py')

sampling_mode = 'GreyST'
greyscale=True

def _dataloader_shape_to_model_input_shape(inputs):
    N, C, T, H, W = inputs.shape        # C = 1
    GreyST = inputs.view(N,3,T//3,H,W).reshape(N,-1,H,W).permute(0,2,1,3,4)
    return model_cfg.NCTHW_to_model_input_shape(GreyST)

