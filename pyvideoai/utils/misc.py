#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import math
import numpy as np
import os
from datetime import datetime
import torch
from matplotlib import pyplot as plt
from torch import nn


import logging
logger = logging.getLogger(__name__)

def check_nan_losses(loss):
    """
    Determine whether the loss is NaN (not a number).
    Args:
        loss (loss): loss to check whether is NaN.
    """
    if math.isnan(loss):
        raise RuntimeError("ERROR: Got NaN losses {}".format(datetime.now()))


def params_count(model):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    return np.sum([p.numel() for p in model.parameters()]).item()


def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (MiB).
    """
    mem_usage_bytes = torch.cuda.max_memory_allocated()
    return mem_usage_bytes / (1024 * 1024)


def log_model_info(model):
    """
    Log model info, includes number of parameters, gpu usage.
    Args:
        model (model): model to log the info.

    Returns:
        str: model info
    """
    logger.info("Model:\n{}".format(model))
    logger.info("Params: {:,}".format(params_count(model)))
    logger.info("Mem: {:,} MiB".format(gpu_mem_usage()))
    logger.info("nvidia-smi")
    logger.info(os.popen("nvidia-smi").read())


def model_info(model):
    """
    Return model info, includes number of parameters, gpu usage.
    Args:
        model (model): model to log the info.

    Returns:
        str: model info
    """
    str_model_info = "Model:\n{}\n".format(model)
    str_model_info += "Params: {:,}\n".format(params_count(model))
    str_model_info += "Mem: {:,} MiB\n".format(gpu_mem_usage())
    str_model_info += "nvidia-smi\n"
    str_model_info += os.popen("nvidia-smi").read()
    return str_model_info


def is_eval_epoch(cfg, cur_epoch):
    """
    Determine if the model should be evaluated at the current epoch.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        cur_epoch (int): current epoch.
    """
    return (
        cur_epoch + 1
    ) % cfg.TRAIN.EVAL_PERIOD == 0 or cur_epoch + 1 == cfg.SOLVER.MAX_EPOCH


def plot_input(tensor, bboxes=(), texts=(), path="./tmp_vis.png"):
    """
    Plot the input tensor with the optional bounding box and save it to disk.
    Args:
        tensor (tensor): a tensor with shape of `NxCxHxW`.
        bboxes (tuple): bounding boxes with format of [[x, y, h, w]].
        texts (tuple): a tuple of string to plot.
        path (str): path to the image to save to.
    """
    tensor = tensor - tensor.min()
    tensor = tensor / tensor.max()
    f, ax = plt.subplots(nrows=1, ncols=tensor.shape[0], figsize=(50, 20))
    for i in range(tensor.shape[0]):
        ax[i].axis("off")
        ax[i].imshow(tensor[i].permute(1, 2, 0))
        # ax[1][0].axis('off')
        if bboxes is not None and len(bboxes) > i:
            for box in bboxes[i]:
                x1, y1, x2, y2 = box
                ax[i].vlines(x1, y1, y2, colors="g", linestyles="solid")
                ax[i].vlines(x2, y1, y2, colors="g", linestyles="solid")
                ax[i].hlines(y1, x1, x2, colors="g", linestyles="solid")
                ax[i].hlines(y2, x1, x2, colors="g", linestyles="solid")

        if texts is not None and len(texts) > i:
            ax[i].text(0, 0, texts[i])
    f.savefig(path)


def frozen_bn_stats(model):
    """
    Set all the bn layers to eval mode.
    Args:
        model (model): model to set bn layers to eval mode.
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eval()

def data_to_gpu(inputs, uids, labels):
    cur_device = torch.cuda.current_device()
    inputs = inputs.to(cur_device, non_blocking=True)
    labels = labels.to(cur_device, non_blocking=True)
    uids = uids.to(cur_device, non_blocking=True)
    curr_batch_size = torch.LongTensor([labels.shape[0]]).to(cur_device, non_blocking=True)

    return inputs, uids, labels, curr_batch_size


def has_gotten_lower(values: list, allow_same: bool = True, EPS: float = 1e-6) -> bool:
    """
    Check if any of the elements is lower than the first element.
    Used for early stopping.

    ```
    if not has_gotten_lower(val_loss[-20:]):
        logger.info("Validation loss hasn't decreased for 20 epochs. Stopping training..")
        raise Exception("Early stopping triggered.")
    ```
    """

    values = [value for value in values if value is not None and not math.isnan(value)]

    if len(values) <= 1:
        raise ValueError("Can't determine if values got lower with 0 or 1 value.")

    if allow_same:
        for value in values[1:]:
            if values[0] > value - EPS:
                return True
    else:
        for value in values[1:]:
            if values[0] > value + EPS:
                return True

    return False


def has_gotten_higher(values: list, allow_same: bool = True, EPS: float = 1e-6) -> bool:
    """
    Check if any of the elements is higher than the first element.
    Used for early stopping.
    If the list contains None, ignore them.

    ```
    if not has_gotten_higher(val_acc[-20:]):
        logger.info("Validation accuracy hasn't increased for 20 epochs. Stopping training..")
        raise Exception("Early stopping triggered.")
    ```
    """

    values = [value for value in values if value is not None and not math.isnan(value)]

    if len(values) <= 1:
        raise ValueError("Can't determine if values got higher with 0 or 1 value.")

    if allow_same:
        for value in values[1:]:
            if values[0] < value + EPS:
                return True
    else:
        for value in values[1:]:
            if values[0] < value - EPS:
                return True

    return False


def has_gotten_better(values: list, is_better_func: callable, allow_same: bool = True) -> bool:
    """
    Check if any of the elements is better than the first element.
    Used for early stopping.
    If the list contains None, ignore them.

    ```
    if not has_gotten_better(val_acc[-20:], is_better_func=lambda a,b: a>b):
        logger.info("Validation accuracy hasn't increased for 20 epochs. Stopping training..")
        raise Exception("Early stopping triggered.")
    ```
    """

    values = [value for value in values if value is not None and not math.isnan(value)]

    if len(values) <= 1:
        raise ValueError("Can't determine if values got better with 0 or 1 value.")

    if allow_same:
        for value in values[1:]:
            if not is_better_func(values[0], value):
                return True
    else:
        for value in values[1:]:
            if is_better_func(value, values[0]):
                return True

    return False
