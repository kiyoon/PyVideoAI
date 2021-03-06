import os
_SCRIPT_DIR = os.path.dirname(os.path.abspath( __file__ ))
exec(open(f'{_SCRIPT_DIR}/../sparsesample_OrderedTCswap_crop224_8frame_largejit_plateau_10scrop.py').read())

input_frame_length = 32
T = input_frame_length
if T % 3 == 0:
    ordering = [(x*(T//3) + x//3) % T for x in range(T)]
elif T % 3 == 1:
    ordering = [(x*((2*T+1)//3)) % T for x in range(T)]
else:
    ordering = [(x*((T+1)//3)) % T for x in range(T)]

def batch_size():
    '''batch_size can be either integer or function returning integer.
    '''
    vram = torch.cuda.get_device_properties(0).total_memory
    if vram > 20e+9:
        return 12
    elif vram > 10e+9:
        return 6
    return 3
