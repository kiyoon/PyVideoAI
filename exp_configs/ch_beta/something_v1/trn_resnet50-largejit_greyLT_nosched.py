import os

from pyvideoai.dataloaders.frames_sparsesample_dataset_advanced import FramesSparsesampleDatasetAdvanced

import torch

batch_size = 8  # per process (per GPU)

input_frame_length = 8
crop_size = 224
train_jitter_min = 224
train_jitter_max = 336
val_scale = 256
val_num_spatial_crops = 1
test_scale = 256
test_num_spatial_crops = 5

def optimiser(params):
    return torch.optim.SGD(params, lr = 0.0001, momentum = 0.9, weight_decay = 5e-4)

def scheduler(optimiser, iters_per_epoch, last_epoch=-1):
    #return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimiser, T_0 = 100 * iters_per_epoch, T_mult = 1, last_epoch=last_epoch)     # Here, last_epoch means last iteration.
    #return torch.optim.lr_scheduler.StepLR(optimiser, step_size = 50 * iters_per_epoch, gamma = 0.1, last_epoch=last_epoch)     # Here, last_epoch means last iteration.
    return None

def load_model():
    return model_cfg.load_model(dataset_cfg.num_classes, input_frame_length)

def load_pretrained(model):
    return

def _dataloader_shape_to_model_input_shape(inputs):
    N, C, T, H, W = inputs.shape        # C = 1
    return inputs.view((N,3,T//3,H,W)).permute((0,2,1,3,4)).reshape((N,-1,H,W))

def get_input_reshape_func(split):
    '''
    if split == 'train':
        return _dataloader_shape_to_model_input_shape
    elif split == 'val':
        return _dataloader_shape_to_model_input_shape
    elif split == 'multicropval':
        return _dataloader_shape_to_model_input_shape
    else:
        assert False, 'unknown split'
    '''
    return _dataloader_shape_to_model_input_shape


def _unpack_data(data):
    inputs, uids, labels, spatial_idx, _, _ = data
    return inputs, uids, labels, spatial_idx


def get_data_unpack_func(split):
    '''
    if split == 'train':
        return _unpack_data
    elif split == 'val':
        return _unpack_data
    elif split == 'multicropval':
        return _unpack_data
    else:
        assert False, 'unknown split'
    '''
    return _unpack_data


def _get_torch_dataset(csv_path, split):
    mode = dataset_cfg.split2mode[split]
    if split == 'val':
        _test_scale = val_scale
        _test_num_spatial_crops = val_num_spatial_crops
    else:
        _test_scale = test_scale
        _test_num_spatial_crops = test_num_spatial_crops
    return FramesSparsesampleDatasetAdvanced(csv_path, mode,
            input_frame_length*3, 
            train_jitter_min = train_jitter_min, train_jitter_max=train_jitter_max,
            test_scale = _test_scale, test_num_spatial_crops=_test_num_spatial_crops,
            crop_size=crop_size,
            mean = [model_cfg.input_mean[0]], std = [model_cfg.input_std[0]],
            normalise = model_cfg.input_normalise, bgr=model_cfg.input_bgr,
            greyscale = True,
            path_prefix=dataset_cfg.frames_dir)

def get_torch_dataset(split):

    mode = dataset_cfg.split2mode[split]
    csv_path = os.path.join(dataset_cfg.frames_split_file_dir, dataset_cfg.split_file_basename[split])

    return _get_torch_dataset(csv_path, split)
