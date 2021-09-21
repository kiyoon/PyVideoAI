_exec_relative_('../sparsesample_RGB_crop224_8frame_largejit_plateau_10scrop.py')

input_frame_length = 24

def batch_size():
    '''batch_size can be either integer or function returning integer.
    '''
    devices=list(range(torch.cuda.device_count()))
    vram = min([torch.cuda.get_device_properties(device).total_memory for device in devices])
    if vram > 20e+9:
        return 8
    elif vram > 10e+9:
        return 4
    return 2
