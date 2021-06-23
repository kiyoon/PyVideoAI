import os

from ...dataloaders.frames_sparsesample_dataset import FramesSparsesampleDataset
from ...dataloaders.frames_sparsesample_dataset_balanced import FramesSparsesampleDatasetBalanced

import torch
import numpy as np
import logging
from torch.utils.data.distributed import DistributedSampler
logger = logging.getLogger(__name__)

input_frame_length = 8

def criterion():
    return torch.nn.CrossEntropyLoss()

def epoch_start_script(epoch, exp, args, rank, world_size, train_kit):
    if epoch == 30:
        logger.info("applying balanced class weight and freezing the base model")
        csv_path = os.path.join(dataset_cfg.frames_split_file_dir, dataset_cfg.split_file_basename['train'])
        dataset = FramesSparsesampleDatasetBalanced(csv_path, input_frame_length, dataset_cfg.num_classes, num_samples_per_class=100,
                frame_start_idx=0, num_zero_pad_in_filenames=5,
                train_jitter_min = 256, train_jitter_max=320,
                mean = model_cfg.input_mean, std = model_cfg.input_std,
                normalise = model_cfg.input_normalise, bgr=model_cfg.input_bgr)
        train_sampler = DistributedSampler(dataset) if world_size > 1 else None
        train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False if train_sampler else True, sampler=train_sampler, num_workers=args.dataloader_num_workers, pin_memory=True, drop_last=True)
        iters_per_epoch = len(train_dataloader)
        last_iter_num = epoch * iters_per_epoch 
        train_kit["scheduler"] = scheduler(train_kit["optimiser"], iters_per_epoch, last_epoch=last_iter_num-1)   # the last_epoch always has to be {the number of step() called} - 1
        train_kit["train_dataloader"] = train_dataloader
        # freeze base
        if world_size > 1:
            for param in train_kit["model"].module.base_model.parameters():
                param.requires_grad = False
        else:
            for param in train_kit["model"].base_model.parameters():
                param.requires_grad = False

def optimiser(params):
    return torch.optim.SGD(params, lr = 0.001, momentum = 0.9, weight_decay = 5e-4)

def scheduler(optimiser, iters_per_epoch, last_epoch=-1):
    #return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimiser, T_0 = 100 * iters_per_epoch, T_mult = 1, last_epoch=last_epoch)     # Here, last_epoch means last iteration.
    return torch.optim.lr_scheduler.StepLR(optimiser, step_size = 30 * iters_per_epoch, gamma = 0.01, last_epoch=last_epoch)     # Here, last_epoch means last iteration.

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
            input_frame_length, frame_start_idx=0, num_zero_pad_in_filenames=5, test_num_spatial_crops=5,
            train_jitter_min = 256, train_jitter_max=320,
            mean = model_cfg.input_mean, std = model_cfg.input_std,
            normalise = model_cfg.input_normalise, bgr=model_cfg.input_bgr)

def _get_torch_dataset_split(split):

    mode = dataset_cfg.split2mode[split]
    csv_path = os.path.join(dataset_cfg.frames_split_file_dir, dataset_cfg.split_file_basename[split])

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
