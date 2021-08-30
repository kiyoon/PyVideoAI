_exec_relative_('sparsesample_video_RGB.py')

sampling_mode = 'GreyST'
greyscale=True

def _dataloader_shape_to_model_input_shape(inputs):
    N, C, T, H, W = inputs.shape        # C = 1
    return inputs.view((N,3,T//3,H,W)).reshape((N,-1,H,W))

