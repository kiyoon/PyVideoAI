import os
_SCRIPT_DIR = os.path.dirname(os.path.abspath( __file__ ))
exec(open(f'{_SCRIPT_DIR}/../i3d_resnet50-crop224_lr0001_8x8_largejit_plateau_1scrop5tcrop.py').read())

batch_size = 1

def optimiser(params):
    return torch.optim.SGD(params, lr = 0.008, momentum = 0.9, weight_decay = 5e-4)

def scheduler(optimiser, iters_per_epoch, last_epoch=-1):
    return None
