#!/usr/bin/env python3
'''
model_cfg needs NCTHW_to_model_input_shape(inputs) function
'''
import argparse
import logging
import numpy as np
import os
import sys

import coloredlogs, verboselogs
logger = verboselogs.VerboseLogger(__name__)

import torch
from torch import nn
from experiment_utils import ExperimentBuilder
import model_configs
import time

from pyvideoai import config
from pyvideoai.utils import misc

from fvcore.nn.activation_count import activation_count
from fvcore.nn.flop_count import FlopCountAnalysis

def main():
    parser = argparse.ArgumentParser(
        description='Randomly sample some training videos and send to Telegram/TensorBoard as GIF. Run with no GPUs (CUDA_VISIBLE_DEVICES=)')

    parser.add_argument("-M", "--model", type=str, default='tsn_resnet50', help="Model")
    parser.add_argument("-c:m", "--model_channel", type=str, default='', choices = model_configs.available_channels, help="Model channel")
    parser.add_argument("--seed", type=int, default=12, help="Random seed for np, torch, torch.cuda, DALI. Actual seed will be seed+rank.")
    parser.add_argument("--mode", type=str, default='flop', choices=['flop', 'activation'],  help="Giga flop or mega activation count.")
    parser.add_argument("--crop_size", type=int, default=224, help="Input crop size")
    parser.add_argument("--num_classes", type=int, default=51, help="Input crop size")
    parser.add_argument("--sequence_length", type=int, default=8, help="Input #frame.")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    coloredlogs.install(fmt='%(name)s: %(lineno)4d - %(levelname)s - %(message)s', level='INFO')

    model_cfg = model_configs.load_cfg(args.model, args.model_channel)

    try:
        # Network
        # Construct the model
        model = model_cfg.load_model(args.num_classes, args.sequence_length)
        misc.log_model_info(model)


        N = 1
        C = 3
        T = args.sequence_length
        H = args.crop_size
        W = args.crop_size
        dummy_input = torch.rand(N, C, T, H, W)
        inputs = model_cfg.NCTHW_to_model_input_shape(dummy_input)

        if args.mode == "flop":
            flops = FlopCountAnalysis(model, inputs)
            count = flops.total()
            logger.info(f"Flop count: {count / 1e+9} G")
            print(flops.by_module())
        elif args.mode == "activation":
            count_dict, *_ = activation_count(model, inputs)
            count = sum(count_dict.values())
            logger.info(f"Activation count: {count} M")

    except Exception as e:
        logger.exception("Exception occurred")



if __name__ == '__main__':
    main()
