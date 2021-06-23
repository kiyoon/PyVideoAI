import os

from ...dataloaders.frames_sparsesample_dataset import FramesSparsesampleDataset

import torch
from ...utils.lr_scheduling import multistep_lr_nonlinear_iters

input_frame_length = 8


def optimiser(params):
    return torch.optim.SGD(params, lr = 0.0001, momentum = 0.9, weight_decay = 5e-4)

def scheduler(optimiser, iters_per_epoch, last_epoch=-1):
    #return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimiser, T_0 = 100 * iters_per_epoch, T_mult = 1, last_epoch=last_epoch)     # Here, last_epoch means last iteration.
    #return torch.optim.lr_scheduler.StepLR(optimiser, step_size = 50 * iters_per_epoch, gamma = 0.1, last_epoch=last_epoch)     # Here, last_epoch means last iteration.
    return multistep_lr_nonlinear_iters(optimiser, iters_per_epoch, [35, 50, 100], [10, 20, 50], last_epoch=last_epoch)
    #return None

def load_model():
    return model_cfg.load_model(dataset_cfg.num_classes, input_frame_length)


#def model_predict(model, inputs):
#    verb_outputs = model(inputs)
#    return verb_outputs

def model_predict(model, inputs):
    N, C, T, H, W = inputs.shape
    
    # N, C, T, H, W -> N, T, C, H, W -> N, TC, H, W
    #verb_outputs, noun_outputs = model(inputs.permute(0,2,1,3,4).reshape((N, -1, H, W)))
    verb_outputs = model(inputs.permute(0,2,1,3,4).reshape((N, -1, H, W)))
    return verb_outputs


def unpack_data(data):
    inputs, uids, labels, spatial_idx, _, _, _ = data
    return dataset_cfg._data_to_gpu(inputs, uids, labels) + (spatial_idx,)
    #inputs, labels = data
    #return dataset_cfg._data_to_gpu(inputs, labels, labels) + (labels,)

def get_torch_dataset(csv_path, mode):
    return FramesSparsesampleDataset(csv_path, mode,
            input_frame_length, frame_start_idx=0, test_num_spatial_crops=5,
            train_jitter_min = 224, train_jitter_max=336,
            path_prefix=dataset_cfg.image_frames_dir,
            mean = model_cfg.input_mean, std = model_cfg.input_std,
            normalise = model_cfg.input_normalise, bgr=model_cfg.input_bgr)

def _get_torch_dataset_split(split):

    mode = dataset_cfg.split2mode[split]
    csv_path = os.path.join(dataset_cfg.frames_split_partial_file_dir, dataset_cfg.split_file_basename[split])

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


'''
def get_torch_datasets(splits=['train', 'val', 'multicropval']):
    if type(splits) == list:
        ret_datasets = {}
        for split in splits:
            ret_datasets[split] = get_torch_datasets(split)
        return ret_datasets


    elif type(splits) == str:
        scale_size = 256
        crop_size = 224
        categories, train_list, val_list, root_path, prefix = return_dataset("something", "RGB")

        if splits == 'train':
            train_augmentation = torchvision.transforms.Compose([GroupMultiScaleCrop(crop_size, [1, .875, .75, .66]),
                                               GroupRandomHorizontalFlip(is_flow=False)])
            return TSNDataSet(root_path, train_list, num_segments=input_frame_length,
                       new_length=1,
                       modality="RGB",
                       image_tmpl=prefix,
                       transform=torchvision.transforms.Compose([
                           train_augmentation,
                           Stack(roll=True),
                           ToTorchFormatTensor(div=False),
                           GroupNormalize(model_cfg.input_mean, model_cfg.input_std)
                       ]))


        elif splits == 'val':
            return TSNDataSet(root_path, val_list, num_segments=input_frame_length,
                           new_length=1,
                           modality="RGB",
                           image_tmpl=prefix,
                           random_shift=False,
                           transform=torchvision.transforms.Compose([
                               GroupScale(int(scale_size)),
                               GroupCenterCrop(crop_size),
                               Stack(roll=True),
                               ToTorchFormatTensor(div=False),
                               GroupNormalize(model_cfg.input_mean, model_cfg.input_std)
                           ]))
        else:
            raise ValueError('Unsupported dataset split type: {}'.format(splits))
    else:
        raise ValueError('Only list and str is compatibile for the splits')
'''
