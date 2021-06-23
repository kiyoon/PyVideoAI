import os

from ...dataloaders.frames_densesample_dataset import FramesDensesampleDataset

import torch

batch_size = 4  # per process (per GPU)

input_frame_length = 8
input_sample_rate = 8
crop_size = 112
train_jitter_min = 112
train_jitter_max = 168
val_scale = 112
val_num_ensemble_views = 1
val_num_spatial_crops = 1
test_scale = 112
test_num_ensemble_views = 5
test_num_spatial_crops = 1

input_channel_num=[3]   # RGB

def optimiser(params):
    return torch.optim.SGD(params, lr = 0.001, momentum = 0.9, weight_decay = 5e-4)

def scheduler(optimiser, iters_per_epoch, last_epoch=-1):
    #return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimiser, T_0 = 100 * iters_per_epoch, T_mult = 1, last_epoch=last_epoch)     # Here, last_epoch means last iteration.
    return torch.optim.lr_scheduler.StepLR(optimiser, step_size = 50 * iters_per_epoch, gamma = 0.1, last_epoch=last_epoch)     # Here, last_epoch means last iteration.
    #return None

def load_model():
    return model_cfg.load_model(dataset_cfg.num_classes, input_frame_length, crop_size, input_channel_num)

def load_pretrained(model):
    model_cfg.load_pretrained(model, model_cfg.pretrained_path_8x8)

def model_predict(model, inputs):
    return model_cfg.model_predict_NCTHW(model, inputs)


def unpack_data(data):
    inputs, uids, labels, spatial_idx, temporal_idx, _, _ = data
    return dataset_cfg._data_to_gpu(inputs, uids, labels) + ({"spatial_idx": spatial_idx, "temporal_idx": temporal_idx},)

def get_torch_dataset(csv_path, split):
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

def _get_torch_dataset_split(split):

    csv_path = os.path.join(dataset_cfg.frames_split_file_dir, dataset_cfg.split_file_basename[split])

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

