import os
_SCRIPT_DIR = os.path.dirname(os.path.abspath( __file__ ))
exec(open(f'{_SCRIPT_DIR}/../sparsesample_RGB_crop224_8frame_largejit_plateau_10scrop.py').read())

input_frame_length = 32
def batch_size():
    '''batch_size can be either integer or function returning integer.
    '''
    vram = torch.cuda.get_device_properties(0).total_memory
    if vram > 20e+9:
        return 12
    elif vram > 10e+9:
        return 6
    return 3
