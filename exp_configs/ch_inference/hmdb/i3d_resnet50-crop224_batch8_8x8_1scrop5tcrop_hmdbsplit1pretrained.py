import os

from pyvideoai.dataloaders.frames_densesample_dataset import FramesDensesampleDataset

import torch

#batch_size = 8  # per process (per GPU)
def batch_size():
    '''batch_size can be either integer or function returning integer.
    '''
    vram = cuda.get_device_properties(0).total_memory
    if vram > 20e+9:
        return 32
    elif vram > 10e+9:
        return 16
    return 8

input_frame_length = 8
input_sample_rate = 8
crop_size = 224
train_jitter_min = 224
train_jitter_max = 336
val_scale = 224
val_num_ensemble_views = 1
val_num_spatial_crops = 1
test_scale = 224
test_num_ensemble_views = 5
test_num_spatial_crops = 1

input_channel_num=[3]   # RGB


#### OPTIONAL
## when you resume from checkpoint, load optimiser/scheduler state?
## Default values are True.
#load_optimiser_state = True
#load_scheduler_state = True
#
#def criterion():
#    return torch.nn.CrossEntropyLoss()
#
#def epoch_start_script(epoch, exp, args, rank, world_size, train_kit):
#   return None
#
#def get_optim_policies(model):
#    """
#    You can set different learning rates on different blocks of the network.
#    Refer to `get_optim_policies()` in pyvideoai/models/epic/tsn.py
#    """
#    conv_weight = []
#    conv_bias = []
#    for m in model.parameters():
#        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
#            ps = list(m.parameters())
#            conv_weight.append(ps[0])
#            if len(ps) == 2:
#                conv_bias.append(ps[1])
#        # ...
#    return [
#        {   
#            "params": conv_weight,
#            "lr_mult": 1,
#            "decay_mult": 1,
#            "name": "conv_weight",
#        },
#        {   
#            "params": conv_bias,
#            "lr_mult": 2,
#            "decay_mult": 0,
#            "name": "conv_bias",
#        },
#    ]
#
#import logging
#logger = logging.getLogger(__name__)
#from pyvideoai.utils.misc import has_gotten_lower, has_gotten_higher
## optional
#def early_stopping_condition(epoch, exp):
#    patience=20
#    if epoch+1 >= patience:
#        if not has_gotten_lower(exp.summary['val_loss'][-patience:]) and not has_gotten_higher(exp.summary['val_acc'][-patience:]):
#            logger.info(f"Validation loss and accuracy haven't gotten better for {patience} epochs. Stopping training..")
#            return True
#
#    return False

def optimiser(params):
    return torch.optim.SGD(params, lr = 0.0001, momentum = 0.9, weight_decay = 5e-4)

def scheduler(optimiser, iters_per_epoch, last_epoch=-1):
    #return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimiser, T_0 = 100 * iters_per_epoch, T_mult = 1, last_epoch=last_epoch)     # Here, last_epoch means last iteration.
    #return torch.optim.lr_scheduler.StepLR(optimiser, step_size = 50 * iters_per_epoch, gamma = 0.1, last_epoch=last_epoch)     # Here, last_epoch means last iteration.
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min', factor=0.1, patience=10, verbose=True)     # NOTE: This special scheduler will ignore iters_per_epoch and last_epoch.
    #return None

def load_model():
    return model_cfg.load_model(dataset_cfg.num_classes, input_frame_length, crop_size, input_channel_num)

# optional
def load_pretrained(model):
    model_cfg.load_pretrained_hmdb(model)

def _dataloader_shape_to_model_input_shape(inputs):
    return model_cfg.NCTHW_to_model_input_shape(inputs)

def get_input_reshape_func(split):
    '''
    if split == 'train':
        return _dataloader_shape_to_model_input_shape
    elif split == 'val':
        return _dataloader_shape_to_model_input_shape
    elif split == 'multicropval':
        return _dataloader_shape_to_model_input_shape
    else:
        raise ValueError(f'Unknown split: {split}')
    '''
    return _dataloader_shape_to_model_input_shape


def _unpack_data(data):
    '''
    From dataloader returning values to (inputs, uids, labels, [reserved]) format
    '''
    inputs, uids, labels, spatial_idx, temporal_idx, _, _ = data
    return inputs, uids, labels, {"spatial_idx": spatial_idx, "temporal_idx": temporal_idx}

def get_data_unpack_func(split):
    '''
    if split == 'train':
        return _unpack_data
    elif split == 'val':
        return _unpack_data
    elif split == 'multicropval':
        return _unpack_data
    else:
        raise ValueError(f'Unknown split: {split}')
    '''
    return _unpack_data

def _get_torch_dataset(csv_path, split):
    mode = dataset_cfg.split2mode[split]

    if split == 'val':
        _test_scale = val_scale
        _test_num_ensemble_views = val_num_ensemble_views
        _test_num_spatial_crops = val_num_spatial_crops
    else:
        _test_scale = test_scale
        _test_num_ensemble_views = test_num_ensemble_views
        _test_num_spatial_crops = test_num_spatial_crops
    return FramesDensesampleDataset(csv_path, mode,
            input_frame_length, input_sample_rate,
            train_jitter_min = train_jitter_min, train_jitter_max=train_jitter_max,
            test_scale=_test_scale, test_num_ensemble_views=_test_num_ensemble_views, test_num_spatial_crops=_test_num_spatial_crops,
            crop_size=crop_size,
            mean = model_cfg.input_mean, std = model_cfg.input_std,
            normalise = model_cfg.input_normalise, bgr=model_cfg.input_bgr,
            path_prefix=dataset_cfg.frames_dir)

def get_torch_dataset(split):

    csv_path = os.path.join(dataset_cfg.frames_split_file_dir, dataset_cfg.split_file_basename1[split])

    return _get_torch_dataset(csv_path, split)
