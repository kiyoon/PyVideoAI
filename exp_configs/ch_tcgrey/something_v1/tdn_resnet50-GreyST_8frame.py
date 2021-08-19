import os
_SCRIPT_DIR = os.path.dirname(os.path.abspath( __file__ ))
exec(open(f'{_SCRIPT_DIR}/../sparsesample_RGB_crop224_8frame_largejit_plateau_10scrop.py').read())  # NOTE: use RGB version, and change sample_index_code to `TDN_GreyST`

greyscale=True
sample_index_code = 'TDN_GreyST'
clip_grad_max_norm = 20

base_learning_rate = 0.01 / 64 / 8     # when batch_size == 1 and #GPUs == 1

def scheduler(optimiser, iters_per_epoch, last_epoch=-1):
    return torch.optim.lr_scheduler.MultiStepLR(optimiser, [it*iters_per_epoch for it in [30, 45, 55]], 0.1, last_epoch)
