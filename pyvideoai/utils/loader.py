#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Data loader."""

import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

import torch.distributed as dist

import logging
logger = logging.getLogger(__name__)

def shuffle_dataset(loader, cur_epoch, seed):
    """"
    Shuffles the data.
    Args:
        loader (loader): data loader to perform shuffle.
        cur_epoch (int): number of the current epoch.
        seed (int): Random seed, but the actual seed would be cur_epoch+seed
    """
    assert isinstance(
        loader.sampler, (RandomSampler, DistributedSampler)
    ), "Sampler type '{}' not supported".format(type(loader.sampler))
    # RandomSampler handles shuffling automatically
    if isinstance(loader.sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        loader.sampler.set_epoch(cur_epoch+seed)


def model_load_state_dict_partial(model, weights_state_dict):
    """https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113
    Load weights partially.
    This ignores the unused parameters in the weights and load the matching keys only.
    """

    model_dict = model.state_dict()

    for k in weights_state_dict:
        if k not in model_dict:
            logger.warning(f'Skip loading parameter: {k}, not in model definition.')
    for k in model_dict:
        if k not in weights_state_dict:
            logger.warning(f'Skip loading module: {k}, not in weight state dict.')

    # 1. filter out unnecessary keys
    weights_state_dict_filtered = {k: v for k, v in weights_state_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(weights_state_dict_filtered) 
    # 3. load the new state dict
    model.load_state_dict(model_dict)

def model_load_state_dict_nostrict(model, weights_state_dict, partial=True):
    """Ignore size mismatch
    https://github.com/PyTorchLightning/pytorch-lightning/issues/4690
    """
    model_state_dict = model.state_dict()

    is_changed = False
    for k in weights_state_dict:
        if k in model_state_dict:
            if weights_state_dict[k].shape != model_state_dict[k].shape:
                logger.warning(f"Skip loading parameter: {k}, "
                        f"required shape: {model_state_dict[k].shape}, "
                        f"loaded shape: {weights_state_dict[k].shape}")
                weights_state_dict[k] = model_state_dict[k]
                is_changed = True
        else:
            #logger.warning(f"Dropping parameter {k}")
            is_changed = True

    if is_changed:
    #   checkpoint.pop("optimizer_states", None) 
        logger.warning("Parameters have been changed. You may not want to use optimiser_state from the checkpoint")

    if partial:
        model_load_state_dict_partial(model, weights_state_dict)
    else:
        model.load_state_dict(weights_state_dict)

    return is_changed


def model_load_weights_GPU(model, weights_path, cur_device = None, world_size = None, model_state_key='model_state'):
    logger.info("Loading weights: " + weights_path) 

    if cur_device is None:
        cur_device = torch.cuda.current_device()
    if world_size is None:
        world_size = dist.get_world_size() if dist.is_initialized() else 1

    checkpoint = torch.load(weights_path, map_location = "cuda:{}".format(cur_device))
    ms = model.module if world_size > 1 else model
    ms.load_state_dict(checkpoint[model_state_key])

    return checkpoint

def model_load_weights_GPU_nostrict(model, weights_path, cur_device = None, world_size = None, model_state_key='model_state', partial=True):
    logger.info("Loading weights but ignoring size: " + weights_path) 

    if cur_device is None:
        cur_device = torch.cuda.current_device()
    if world_size is None:
        world_size = dist.get_world_size() if dist.is_initialized() else 1

    checkpoint = torch.load(weights_path, map_location = "cuda:{}".format(cur_device))
    ms = model.module if world_size > 1 else model

    model_load_state_dict_nostrict(ms, checkpoint[model_state_key], partial=partial)

    return checkpoint
