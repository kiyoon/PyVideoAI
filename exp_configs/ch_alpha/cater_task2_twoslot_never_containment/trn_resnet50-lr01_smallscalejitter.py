import os


#import dataset_configs
#import model_configs

#_script_basename = os.path.basename(os.path.abspath( __file__ ))
#_script_basename_wo_ext = os.path.splitext(_script_basename)[0]
#_script_name_split = _script_basename_wo_ext.split('-')
#dataset_cfg = dataset_configs.load_cfg(_script_name_split[0])
#model_cfg = model_configs.load_cfg(_script_name_split[1])

from ...dataloaders.frames_sparsesample_dataset_advanced import FramesSparsesampleDatasetAdvanced

import torch


input_frame_length = 8


def optimiser(params):
    return torch.optim.SGD(params, lr = 0.01, momentum = 0.9, weight_decay = 5e-4)

def scheduler(optimiser, iters_per_epoch, last_epoch=-1):
    #return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimiser, T_0 = 100 * iters_per_epoch, T_mult = 1, last_epoch=last_epoch)     # Here, last_epoch means last iteration.
    return torch.optim.lr_scheduler.StepLR(optimiser, step_size = 50 * iters_per_epoch, gamma = 0.1, last_epoch=last_epoch)     # Here, last_epoch means last iteration.

def load_model():
    return model_cfg.load_model(dataset_cfg.num_classes, input_frame_length)


def model_predict(model, inputs):
    return model(inputs)


def unpack_data(data):
    inputs, uids, labels, spatial_idx, _, _ = data
    return dataset_cfg._data_to_gpu(inputs, uids, labels) + (spatial_idx,)

def get_torch_dataset(csv_path, mode):
    return FramesSparsesampleDatasetAdvanced(csv_path, mode,
            input_frame_length, test_num_spatial_crops=5,
            train_jitter_min = 256, train_jitter_max=320,
            path_prefix=os.path.join(dataset_cfg.dataset_root, "frames"),
            mean = model_cfg.input_mean, std = model_cfg.input_std,
            normalise = model_cfg.input_normalise, bgr=model_cfg.input_bgr)

def _get_torch_dataset_split(split):

    mode = dataset_cfg.split2mode[split]
    csv_path = os.path.join(dataset_cfg.frames_splits_dir, dataset_cfg.split_file_basename[split])

    return get_torch_dataset(csv_path, mode)
    

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
