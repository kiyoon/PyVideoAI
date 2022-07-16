#!/usr/bin/env python3
"""Wrapper to train and test a video classification model."""

import argparse
from pyvideoai.train_multiprocess import train
from experiment_utils.argparse_utils import add_exp_arguments

import dataset_configs
import model_configs
import exp_configs
from pyvideoai import config

def add_train_args(parser):
    parser.add_argument("-e", "--num_epochs", default='DEPRECATED', help="DEPRECATED.")
    add_exp_arguments(parser,
            root_default=config.DEFAULT_EXPERIMENT_ROOT, dataset_default='hmdb', model_default='i3d_resnet50', name_default='crop224_8x8_largejit_plateau_1scrop5tcrop_split1',
            dataset_channel_choices=dataset_configs.available_channels, model_channel_choices=model_configs.available_channels, exp_channel_choices=exp_configs.available_channels)
    parser.add_argument("-s", "--save_mode", type=str, default="last_and_peaks", choices=["all", "higher", "last_and_peaks"],  help="Checkpoint saving condition. all: save all epochs, higher: save whenever the highest validation performance model is found, last_and_peaks: save all epochs, but remove previous epoch if that wasn't the best.")
    parser.add_argument("--training_speed", type=str, default="standard", choices=["standard", "faster"],  help="Only applicable when using distributed multi-GPU training. 'faster' skips multiprocess commuication and CPU-GPU synchronisation for calculating training metrics (loss, accuracy) so they will be reported as 0. This probably won't give you any benefit on a single node, but for multi-node, it depends on the internet connection.")
    parser.add_argument("-l", "--load_epoch", type=int, default=None, help="Load from checkpoint. Set to -1 to load from the last checkpoint.")
    parser.add_argument("--seed", type=int, default=12, help="Random seed for np, torch, torch.cuda, DALI.")
    parser.add_argument("-t", "--multi_crop_val_period", type=int, default=-1, help="Number of epochs after full multi-crop validation is performed.")
    parser.add_argument("-T", "--telegram_post_period", type=int, default=10, help="Period (in epochs) to send the training stats on Telegram.")
    parser.add_argument("-B", "--telegram_bot_idx", type=int, default=0, help="Which Telegram bot to use defined in key.ini?")
    parser.add_argument("-WP", "--wandb_project", help="Weights & Biases (wandb.ai) project name. Assigning this argument enables the Weights & Biases visualisation service. Make sure you install wandb and call `wandb login` before you use.")
    parser.add_argument("-WE", "--wandb_entity", help="Weights & Biases (wandb.ai) entity name.")
    parser.add_argument("-WR", "--wandb_run_id", help="To resume, specify the run ID of Weights & Biases (wandb.ai)")
    parser.add_argument("-WUM", "--wandb_upload_models", default='best_and_last', choices=['best_and_last', 'no'], help="Upload models to Weights & Biases (wandb.ai)")
    parser.add_argument("-w", "--dataloader_num_workers", type=int, default=4, help="num_workers for PyTorch Dataset loader.")
    parser.add_argument("-r", "--refresh_period", type=int, default=1, help="How many iterations until printing stats. Increase this if stdio is your bottleneck (such as Slurm printing to network file).")
    parser.add_argument("-v", "--version", type=str, default='auto', help="Experiment version (`auto` or integer). `auto` chooses the last version when resuming from the last, otherwise creates new version.")
    parser.add_argument("-L", "--console_log_level", type=str, default='INFO', choices=['NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help="Logging level for console output.")



def get_parser():
    parser = argparse.ArgumentParser(description="Train and validate an action model",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_train_args(parser)
    return parser


def main():
    """
    Main function to spawn the train and test process.
    """
    parser = get_parser()
    args = parser.parse_args()

    train(args)



if __name__ == "__main__":
    main()
