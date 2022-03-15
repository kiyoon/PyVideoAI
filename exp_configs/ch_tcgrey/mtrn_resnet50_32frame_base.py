input_frame_length = 32
def batch_size():
    '''batch_size can be either integer or function returning integer.
    '''
    vram = torch.cuda.get_device_properties(0).total_memory
    if vram > 20e+9:
        return 8
    elif vram > 10e+9:
        return 4
    return 2
