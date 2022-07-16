#!/usr/bin/env python3
"""Wrapper to train and test a video classification model."""

import argparse

from pyvideoai.eval_multiprocess import evaluation
from experiment_utils.argparse_utils import add_exp_arguments

import dataset_configs
import model_configs
import exp_configs
from pyvideoai import config

def add_eval_args(parser):
    add_exp_arguments(parser,
            root_default=config.DEFAULT_EXPERIMENT_ROOT, dataset_default='hmdb', model_default='i3d_resnet50', name_default='crop224_8x8_largejit_plateau_1scrop5tcrop_split1',
            dataset_channel_choices=dataset_configs.available_channels, model_channel_choices=model_configs.available_channels, exp_channel_choices=exp_configs.available_channels)
    parser.add_argument("-l", "--load_epoch", type=int, default=None, help="Load from checkpoint. Set to -1 to load from the last checkpoint, and to -2 to load best model in terms of val_acc.")
    parser.add_argument("--seed", type=int, default=12, help="Random seed for np, torch, torch.cuda, DALI.")
    parser.add_argument("-m", "--mode", type=str, default="oneclip", choices=["oneclip", "multicrop"],  help="Evaluate using 1 clip or 30 clips.")
    parser.add_argument("-s", "--split", type=str, default=None,  help="Which split to use to evaluate? Default is auto (None)")
    parser.add_argument("-p", "--save_predictions", action='store_true',  help="Save the predictions after the evaluation.")
    parser.add_argument("-w", "--dataloader_num_workers", type=int, default=4, help="num_workers for PyTorch Dataset loader.")
    parser.add_argument("-B", "--telegram_bot_idx", type=int, default=0, help="Which Telegram bot to use defined in key.ini?")
    parser.add_argument("-r", "--refresh_period", type=int, default=1, help="How many iterations until printing stats. Increase this if stdio is your bottleneck (such as Slurm printing to network file).")
    parser.add_argument("-v", "--version", type=str, default='auto', help="Experiment version (`auto` or integer). `auto` chooses the last version.")
    parser.add_argument("-L", "--console_log_level", type=str, default='INFO', choices=['NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help="Logging level for console output.")
    parser.add_argument("-W", "--wandb", action='store_true', help="Detect Weights & Biases project name and run ID automatically, and update the summary and predictions")
    parser.add_argument("-WE", "--wandb_entity", help="Weights & Biases (wandb.ai) entity name.")



def get_parser():
    parser = argparse.ArgumentParser(description="Evaluate an action model",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_eval_args(parser)
    return parser


def main():
    """
    Main function to spawn the train and test process.
    """
    parser = get_parser()
    args = parser.parse_args()

    evaluation(args)



if __name__ == "__main__":
    main()
