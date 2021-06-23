"""
Evaluate accuracy or mAP using the predictions
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import pickle
import csv
import logging
import numpy as np
import os
import sys


FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


from sklearn.metrics import accuracy_score

import torch
from torch import nn
from experiment_utils.csv_to_dict import csv_to_dict
from experiment_utils.argparse_utils import add_exp_arguments
from experiment_utils import ExperimentBuilder
from pyvideoai import config
import dataset_configs, model_configs
import time


def accuracy(video_labels, video_predictions):
    pred_labels = np.argmax(video_predictions, axis=1)
    return accuracy_score(video_labels, pred_labels)

from utils.AP import mAP

from utils.video_metrics import VideoMetrics

def evaluate_pred(args):
    """evaluate accuracy or mAP"""

    dataset_cfg = dataset_configs.load_cfg(args.dataset)

    perform_multicropval=True       # when loading, assume there was multicropval. Even if there was not, having more CSV field information doesn't hurt.
    if dataset_cfg.task == 'singlelabel_classification':
        multilabel = False
        summary_fieldnames, summary_fieldtypes = ExperimentBuilder.return_fields_singlelabel(multicropval = perform_multicropval)
    elif dataset_cfg.task == 'multilabel_classification':
        multilabel = True
        summary_fieldnames, summary_fieldtypes = ExperimentBuilder.return_fields_multilabel(multicropval = perform_multicropval)
    else:
        raise ValueError()


    def load_predictions(load_epoch, mode, exp, multilabel=False):
        real_load_epoch = None
        if load_epoch == -1:
            real_load_epoch = int(exp.summary['epoch'][-1])
        elif load_epoch == -2:
            if multilabel:
                if mode == 'oneclip':
                    field = 'val_vid_mAP'
                else:   # multicrop
                    field = 'multi_crop_val_vid_mAP'
            else:
                if mode == 'oneclip':
                    field = 'val_acc'
                else:   # multicrop
                    field = 'multi_crop_val_vid_acc_top1'
            real_load_epoch = int(exp.get_best_model_stat(field)['epoch'])
        elif load_epoch >= 0:
            real_load_epoch = load_epoch
        else:
            raise ValueError("Wrong load_epoch value: {:d}".format(load_epoch))

        predictions_file_path = os.path.join(exp.predictions_dir, 'epoch_%04d_%sval.pkl' % (real_load_epoch, mode))
        with open(predictions_file_path, 'rb') as f:
            predictions = pickle.load(f)

        video_predictions = predictions['video_predictions']
        video_labels = predictions['video_labels']
        video_ids = predictions['video_ids']

        return video_predictions, video_labels, video_ids

    ## 1
    exp = ExperimentBuilder(args.experiment_root, args.dataset, args.model, args.experiment_name, summary_fieldnames = summary_fieldnames, summary_fieldtypes = summary_fieldtypes, telegram_key_ini = config.KEY_INI_PATH)
    exp.load_summary()
    video_predictions, video_labels, video_ids = load_predictions(args.load_epoch, args.mode, exp, multilabel)


    ## 2
    exp2 = ExperimentBuilder(args.experiment_root, args.dataset, args.model2, args.experiment_name2, summary_fieldnames = summary_fieldnames, summary_fieldtypes = summary_fieldtypes, telegram_key_ini = config.KEY_INI_PATH)
    exp2.load_summary()
    video_predictions2, video_labels2, video_ids2 = load_predictions(args.load_epoch2, args.mode2, exp2, multilabel)

    '''
    for i in range(10):
        print(video_predictions[i].max())
        print(video_predictions[i].min())
        print(video_labels[i])
        print()
    '''

    video_metrics = VideoMetrics()
    title_str = f'Using {args.dataset} dataset.'
    print(title_str)
    if multilabel:
        print_str = f'mAP for {args.model} {args.experiment_name}: {mAP(video_labels, video_predictions):.4f}\n'
        print_str += f'mAP for {args.model2} {args.experiment_name2}: {mAP(video_labels2, video_predictions2):.4f}\n'
        video_metrics.add_clip_predictions(video_ids, video_predictions, video_labels, 'sigmoid')
        video_metrics.add_clip_predictions(video_ids2, video_predictions2, video_labels2, 'sigmoid')
        print_str += f'mAP for the twostream averaged model: {video_metrics.mAP():.4f}'
    else:
        print_str = f'Accuracy for {args.model} {args.experiment_name}: {accuracy(video_labels, video_predictions):.4f}\n'
        print_str += f'Accuracy for {args.model2} {args.experiment_name2}: {accuracy(video_labels2, video_predictions2):.4f}\n'
        video_metrics.add_clip_predictions(video_ids, video_predictions, video_labels, 'softmax')
        video_metrics.add_clip_predictions(video_ids2, video_predictions2, video_labels2, 'softmax')
        print_str += f'Accuracy for the twostream averaged model: {video_metrics.accuracy()[0]:.4f}'
    print(print_str)
    exp.tg_send_text_with_title(title_str, print_str)


def main():
    parser = argparse.ArgumentParser(
        description='Load predictions, and generating per-class accuracy and confusion matrix')

    add_exp_arguments(parser, dataset_configs.available_datasets, model_configs.available_models, root_default=config.DEFAULT_EXPERIMENT_ROOT, dataset_default='epic_verb', model_default='i3d', name_default='test')
    parser.add_argument("-l", "--load_epoch", type=int, default=-2, help="Load from checkpoint. Set to -1 to load from the last checkpoint, and to -2 to load best model in terms of val_acc.")
    parser.add_argument("-L", "--load_epoch2", type=int, default=-2, help="Load from checkpoint. Set to -1 to load from the last checkpoint, and to -2 to load best model in terms of val_acc.")
    parser.add_argument("-m", "--mode", type=str, default="oneclip", choices=["oneclip", "multicrop"],  help="Mode used for run_val.py")
    parser.add_argument("-d", "--mode2", type=str, default="oneclip", choices=["oneclip", "multicrop"],  help="Mode used for run_val.py")

    parser.add_argument("-O", "--model2", type=str, default="i3d", choices=model_configs.available_models,  help="The second model")
    parser.add_argument("-X", "--experiment_name2", type=str, default="test",  help="The second experiment name")

    args = parser.parse_args()

    evaluate_pred(args)



if __name__ == '__main__':
    main()
