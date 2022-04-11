#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import math
import numpy as np
import os
from datetime import datetime
import torch
from matplotlib import pyplot as plt
from torch import nn


import logging, coloredlogs
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

def data_to_gpu(*args):         #(inputs, uids, labels):
    """
    Get inputs, labels, uids, etc. and copy to GPU.
    It will return inputs, labels, uids, etc. plus current batch size.
    """
    assert len(args) > 0

    cur_device = torch.cuda.current_device()

    gpu_tensors = []
    for arg in args:
        assert arg.shape[0] == args[0].shape[0], f'Different number of batch size found when copying to GPU. {arg.shape[0]} and {args[0].shape[0]}'
        gpu_tensors.append(arg.to(cur_device, non_blocking=True))

    curr_batch_size = torch.LongTensor([args[0].shape[0]]).to(cur_device, non_blocking=True)

    return *gpu_tensors, curr_batch_size


def install_colour_logger(level=logging.INFO, fmt='%(name)s: %(lineno)4d - %(levelname)s - %(message)s'):
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = coloredlogs.ColoredFormatter(fmt)
    console_handler.setFormatter(console_format)

    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)


def install_file_loggers(logs_dir, levels=[logging.NOTSET, logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR], fmt='%(asctime)s - %(name)s: %(lineno)4d - %(levelname)s - %(message)s', file_prefix='train'):
    """Install multiple file loggers per level.
    """
    f_format = logging.Formatter(fmt)
    root_logger = logging.getLogger()

    if isinstance(levels, str):
        levels = [levels]

    for level in levels:
        if isinstance(level, str):
            str_level = level
            level = getattr(logging, level) # convert to integer level
        elif isinstance(level, int):
            str_level = logging.getLevelName(level)
        else:
            raise ValueError(f'Logging level has to be either integer or string, but got {type(level)}.)')

        f_handler = logging.FileHandler(os.path.join(logs_dir, f'{file_prefix}_{str_level}.log'))
        f_handler.setLevel(level)
        f_handler.setFormatter(f_format)

        # Add handlers to the logger
        root_logger.addHandler(f_handler)


def check_pillow_performance():
    try:
        import PIL
    except ImportError:
        logger.error('Pillow not installed.')

    if '.post' not in PIL.__version__:
        logger.warning("Pillow-SIMD not installed. Refer to https://fastai1.fast.ai/performance.html to improve performance. Ignore this message if you're not on x86 platform.")
    else:
        logger.info(f'Pillow-SIMD running with version {PIL.__version__}')

    # check libjpeg-turbo
    from PIL import features, Image
    from packaging import version

    try:    ver = Image.__version__     # PIL >= 7
    except: ver = Image.PILLOW_VERSION  # PIL <  7

    if version.parse(ver) >= version.parse("5.4.0"):
        if features.check_feature('libjpeg_turbo'):
            logger.info("libjpeg-turbo is on. Jpeg encoding/decoding with Pillow will be faster.")
        else:
            logger.warning("libjpeg-turbo is not on. Jpeg encoding/decoding with Pillow will be slow. Refer to https://fastai1.fast.ai/performance.html to improve performance. Ignore this message if you're not on x86 platform.")
    else:
        logger.warning(f"libjpeg-turbo' status can't be derived - need Pillow(-SIMD)? >= 5.4.0 to tell, current version {ver}")



def check_single_label_target_and_convert_to_numpy(target: torch.Tensor) -> np.array:
    """
    Check if a target vector is a correct single-label form.
    It has to be either shape of (N, ) or (N, C) where N=batch size, C=num classes.
    If it's 2-D, then only one has to be value of 1 and others have to be zero.

    Raises:
        ValueError

    Returns:
        None
    """

    target = target.cpu().numpy()

    if target.ndim == 1:
        is_float = target.dtype in (np.float16, np.float32, np.float64)
        for target_value in target:
            if target_value < 0:
                raise ValueError(f'1D target value has to be zero or positive but got a negative number {target_value}.')
            if is_float and not target_value.is_integer():
                raise ValueError(f'target value has to be integer but got a non-integer number {target_value}.')
    elif target.ndim == 2:
        for sample_target in target:
            num_ones = 0
            for target_value in sample_target:
                if target_value not in (1, 0):
                    raise ValueError(f'2D target value has to be zero or one but got {target_value}.')

                num_ones += target_value == 1

            if num_ones != 1:
                raise ValueError(f'{num_ones} "1" value in a single-label target vector. It should only exist one time.')

    else:
        raise ValueError(f'target has to be 1D or 2D tensor but got {target.ndim}-D.')


    return target
