import os
from ...dataloaders.frames_sparsesample_dataset_advanced import FramesSparsesampleDatasetAdvanced

import torch

import numpy as np
import logging
from ...utils.loss import compute_balanced_class_weight_from_csv
logger = logging.getLogger(__name__)

input_frame_length = 32

train_jitter_min = 256
train_jitter_max = 320

val_scale = 256
val_num_spatial_crops = 1
test_scale = 256
test_num_spatial_crops = 5

crop_size = 224

def criterion():
    return torch.nn.BCEWithLogitsLoss()

#def epoch_start_script(epoch, exp, rank, world_size, train_kit):
#    if epoch == 120:
#        logger.info("applying balanced class weight and freezing the base model")
#        csv_path = os.path.join(dataset_cfg.frames_splits_dir, dataset_cfg.split_file_basename['train'])
#        weight = compute_balanced_class_weight_from_csv(csv_path, 2, dataset_cfg.num_classes, ignore_first_row=False).astype(np.float32)
#        cur_device = torch.cuda.current_device()
#        weight = torch.from_numpy(weight).to(device=cur_device, non_blocking=True)
#        train_kit["criterion"] = torch.nn.BCEWithLogitsLoss(weight=weight)
#        # freeze base
#        if world_size > 1:
#            for param in train_kit["model"].module.base_model.parameters():
#                param.requires_grad = False
#        else:
#            for param in train_kit["model"].base_model.parameters():
#                param.requires_grad = False

def optimiser(params):
    return torch.optim.SGD(params, lr = 0.003, momentum = 0.9, weight_decay = 5e-4)

def scheduler(optimiser, iters_per_epoch, last_epoch=-1):
    #return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimiser, T_0 = 100 * iters_per_epoch, T_mult = 1, last_epoch=last_epoch)     # Here, last_epoch means last iteration.
    return torch.optim.lr_scheduler.StepLR(optimiser, step_size = 50 * iters_per_epoch, gamma = 0.1, last_epoch=last_epoch)     # Here, last_epoch means last iteration.

def load_model():
    return model_cfg.load_model(dataset_cfg.num_classes, input_frame_length)


def model_predict(model, inputs):
    N, C, T, H, W = inputs.shape
    
    # N, C, T, H, W -> N, T, C, H, W -> N, TC, H, W
    return model(inputs.permute(0,2,1,3,4).reshape((N, -1, H, W)))


def unpack_data(data):
    inputs, uids, labels, spatial_idx, _, _ = data
    return dataset_cfg._data_to_gpu(inputs, uids, labels) + (spatial_idx,)

def get_torch_dataset(csv_path, split):
    mode = dataset_cfg.split2mode[split]

    if split == 'val':
        _test_scale = val_scale
        _test_num_spatial_crops = val_num_spatial_crops
    else:
        _test_scale = test_scale
        _test_num_spatial_crops = test_num_spatial_crops

    return FramesSparsesampleDatasetAdvanced(csv_path, mode, input_frame_length, 
            train_jitter_min = train_jitter_min, train_jitter_max=train_jitter_max,
            test_scale = _test_scale, test_num_spatial_crops=_test_num_spatial_crops,
            crop_size = crop_size,
            bgr=model_cfg.input_bgr,
            path_prefix=os.path.join(dataset_cfg.dataset_root, "frames"),
            mean = model_cfg.input_mean, std = model_cfg.input_std,
            normalise = model_cfg.input_normalise)

def _get_torch_dataset_split(split):

    csv_path = os.path.join(dataset_cfg.frames_splits_partial_dir, dataset_cfg.split_file_basename[split])

    return get_torch_dataset(csv_path, split)
    

def get_torch_datasets(splits=['train', 'val', 'multicropval']):

    if type(splits) == list:
        torch_datasets = {}
        for split in splits:
            dataset = _get_torch_dataset_split(split)
            torch_datasets[split] = dataset

        return torch_datasets

    elif type(splits) == str:
        return _get_torch_dataset_split(splits)

    else:
        raise ValueError('Wrong type of splits argument. Must be list or string')

