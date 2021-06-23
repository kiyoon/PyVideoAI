#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Data loader."""

import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

import torch.distributed as dist

import logging
logger = logging.getLogger(__name__)

#def construct_loader(cfg, split):
#    """
#    Constructs the data loader for the given dataset.
#    Args:
#        cfg (CfgNode): configs. Details can be found in
#            slowfast/config/defaults.py
#        split (str): the split of the data loader. Options include `train`,
#            `val`, and `test`.
#    """
#    assert split in ["train", "val", "test"]
#    if split in ["train"]:
#        dataset_name = cfg.TRAIN.DATASET
#        batch_size = int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS)
#        shuffle = True
#        drop_last = True
#    elif split in ["val"]:
#        dataset_name = cfg.TRAIN.DATASET
#        batch_size = int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS)
#        shuffle = False
#        drop_last = False
#    elif split in ["test"]:
#        dataset_name = cfg.TEST.DATASET
#        batch_size = int(cfg.TEST.BATCH_SIZE / cfg.NUM_GPUS)
#        shuffle = False
#        drop_last = False
#
#    # Construct the dataset
#    dataset = build_dataset(dataset_name, cfg, split)
#    # Create a sampler for multi-process training
#    sampler = DistributedSampler(dataset) if cfg.NUM_GPUS > 1 else None
#    # Create a loader
#    loader = torch.utils.data.DataLoader(
#        dataset,
#        batch_size=batch_size,
#        shuffle=(False if sampler else shuffle),
#        sampler=sampler,
#        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
#        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
#        drop_last=drop_last,
#        collate_fn=detection_collate if cfg.DETECTION.ENABLE else None,
#    )
#    return loader


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

    # 1. filter out unnecessary keys
    weights_state_dict_filtered = {k: v for k, v in weights_state_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(weights_state_dict_filtered) 
    # 3. load the new state dict
    model.load_state_dict(model_dict)


def model_load_weights_GPU(model, weights_path, cur_device = None, world_size = None):
    logger.info("Loading weights: " + weights_path) 

    if cur_device is None:
        cur_device = torch.cuda.current_device()
    if world_size is None:
        world_size = dist.get_world_size() if dist.is_initialized() else 1

    checkpoint = torch.load(weights_path, map_location = "cuda:{}".format(cur_device))
    ms = model.module if world_size > 1 else model
    ms.load_state_dict(checkpoint["model_state"])

    return checkpoint
