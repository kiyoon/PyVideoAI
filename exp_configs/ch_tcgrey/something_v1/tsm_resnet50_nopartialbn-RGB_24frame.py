import os
_SCRIPT_DIR = os.path.dirname(os.path.abspath( __file__ ))
exec(open(f'{_SCRIPT_DIR}/../sparsesample_RGB_crop224_8frame_largejit_plateau_10scrop.py').read())

input_frame_length = 24
clip_grad_max_norm = 20

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

def optimiser(params):
    """
    LR should be proportional to the total batch size.
    When distributing, LR should be multiplied by the number of processes (# GPUs)
    Thus, LR = base_LR * batch_size_per_proc * (num_GPUs**2)
    """
    batchsize = batch_size() if callable(batch_size) else batch_size
    world_size = get_world_size()

    base_learning_rate = 0.01 / 64     # when batch_size == 1 and #GPUs == 1
    learning_rate = base_learning_rate * batchsize * (world_size**2)

    return torch.optim.SGD(params, lr = learning_rate, momentum = 0.9, weight_decay = 5e-4)


def scheduler(optimiser, iters_per_epoch, last_epoch=-1):
    return torch.optim.lr_scheduler.MultiStepLR(optimiser, [it*iters_per_epoch for it in [20, 40]], 0.1, last_epoch)
