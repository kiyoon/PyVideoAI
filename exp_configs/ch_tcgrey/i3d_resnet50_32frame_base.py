
input_frame_length = 32
base_learning_rate = 1e-4
def batch_size():
    '''batch_size can be either integer or function returning integer.
    '''
    vram = torch.cuda.get_device_properties(0).total_memory
    if vram > 20e+9:
        return 16
    elif vram > 10e+9:
        return 8
    return 4
