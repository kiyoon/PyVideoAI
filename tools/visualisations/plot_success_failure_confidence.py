
import argparse
import os
import pickle

from experiment_utils.argparse_utils import add_exp_arguments
from experiment_utils import ExperimentBuilder

import numpy as np

from pyvideoai import config
import model_configs, dataset_configs, exp_configs

from pyvideoai.utils import misc

import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(description="Read predictions and plot confidence distribution.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_exp_arguments(parser,
            root_default=config.DEFAULT_EXPERIMENT_ROOT, dataset_default='epic100_verb', model_default='tsm_resnet50_nopartialbn', name_default='onehot',
            dataset_channel_choices=dataset_configs.available_channels, model_channel_choices=model_configs.available_channels, exp_channel_choices=exp_configs.available_channels)
    parser.add_argument("-l", "--load_epoch", type=int, default=-2, help="Load from checkpoint. Set to -1 to load from the last checkpoint.")
    parser.add_argument("-m", "--mode", type=str, default="oneclip", choices=["oneclip", "multicrop"],  help="Evaluate using 1 clip or 30 clips.")
    parser.add_argument("-s", "--split", type=str, default=None,  help="Which split to use to evaluate? Default is auto (None)")
    parser.add_argument('-v', '--version', default='last', help='ExperimentBuilder version')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    cfg = exp_configs.load_cfg(args.dataset, args.model, args.experiment_name, args.dataset_channel, args.model_channel, args.experiment_channel)

    metrics = cfg.dataset_cfg.task.get_metrics(cfg)
    summary_fieldnames, summary_fieldtypes = ExperimentBuilder.return_fields_from_metrics(metrics)

    if args.version == 'last':
        _expversion = 'last'    # choose the last version
    else:
        _expversion = int(args.version)


    exp = ExperimentBuilder(args.experiment_root, args.dataset, args.model, args.experiment_name, args.subfolder_name,
            summary_fieldnames = summary_fieldnames, summary_fieldtypes = summary_fieldtypes,
            version=_expversion,
            telegram_key_ini = config.KEY_INI_PATH)

    if args.load_epoch == -1:
        exp.load_summary()
        load_epoch = int(exp.summary['epoch'].iloc[-1])
    elif args.load_epoch == -2:
        exp.load_summary()
        best_metric, best_metric_fieldname = metrics.get_best_metric_and_fieldname()
        logger.info(f'Using the best metric from CSV field `{best_metric_fieldname}`')
        load_epoch = int(exp.get_best_model_stat(best_metric_fieldname, best_metric.is_better)['epoch'])
    elif args.load_epoch >= 0:
        load_epoch = args.load_epoch
    else:
        raise ValueError(f"Wrong args.load_epoch value: {args.load_epoch}")

    # Dataset
    if args.split is not None:
        split = args.split
    else:   # set split automatically
        if args.mode == 'oneclip':
            split = 'val'
        else:   # multicrop
            split = 'multicropval'

    predictions_file_path = os.path.join(exp.predictions_dir, f'epoch_{load_epoch:04d}_{split}_{args.mode}.pkl')

    with open(predictions_file_path, 'rb') as f:
        predictions = pickle.load(f)


    video_predictions = predictions['video_predictions']
    video_labels = predictions['video_labels']
    video_ids = predictions['video_ids']

    video_predlabels = np.argmax(video_predictions, axis=1)
    success_idx = video_predlabels == video_labels
    failure_idx = video_predlabels != video_labels

    video_predictions = video_predictions * 100

    bins = [a* 10 for a in range(11)]
    success_confidence_histogram, _ = np.histogram(video_predictions[success_idx], bins)
    failure_confidence_histogram, _ = np.histogram(video_predictions[failure_idx], bins)
    print(success_confidence_histogram)

    fig = plt.figure()
    ax1 = fig.add_subplot(4, 1, 1)
    ax2 = fig.add_subplot(4, 1, 2)
    ax3 = fig.add_subplot(4, 1, 3)
    ax4 = fig.add_subplot(4, 1, 4)

    ax1.hist(video_predictions[success_idx], bins=bins, density=True, facecolor='b')
    ax2.hist(video_predictions[failure_idx], bins=bins, density=True, facecolor='b')
    ax3.hist(video_predictions[success_idx], bins=bins[1:], density=True, facecolor='b')
    ax4.hist(video_predictions[failure_idx], bins=bins[1:], density=True, facecolor='b')
    #ax1.bar(bins[:-1], success_confidence_histogram)
    #ax2.bar(bins[:-1], failure_confidence_histogram)

    plt.title('Histogram of Success / failure confidences, and without 0-10%')
    ax1.title.set_text('Success')
    ax2.title.set_text('Failure')
    ax3.title.set_text('Success without 0-10%')
    ax4.title.set_text('Failure without 0-10%')



    os.makedirs(exp.plots_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(exp.plots_dir, 'success_failure_confidence_histogram.pdf'))
    fig.savefig(os.path.join(exp.plots_dir, 'success_failure_confidence_histogram.png'))


if __name__ == '__main__':
    main()
