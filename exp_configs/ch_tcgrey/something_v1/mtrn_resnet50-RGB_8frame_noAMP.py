import os
_SCRIPT_DIR = os.path.dirname(os.path.abspath( __file__ ))
exec(open(f'{_SCRIPT_DIR}/../sparsesample_RGB_crop224_8frame_largejit_plateau_10scrop.py').read())

def batch_size():
    '''batch_size can be either integer or function returning integer.
    '''
    vram = torch.cuda.get_device_properties(0).total_memory
    if vram > 20e+9:
        return 16
    elif vram > 10e+9:
        return 8
    return 4

def optimiser(params):
    """
    LR should be proportional to the total batch size.
    When distributing, LR should be multiplied by the number of processes (# GPUs)
    Thus, LR = base_LR * batch_size_per_proc * (num_GPUs**2)
    """
    base_learning_rate = 5e-6      # when batch_size == 1 and #GPUs == 1

    batchsize = batch_size() if callable(batch_size) else batch_size
    world_size = get_world_size()
    learning_rate = base_learning_rate * batchsize * (world_size**2)

    return torch.optim.SGD(params, lr = learning_rate, momentum = 0.9, weight_decay = 5e-4)


use_amp = False
