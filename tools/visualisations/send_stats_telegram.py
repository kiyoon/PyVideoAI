import torch

import os
from experiment_utils import ExperimentBuilder
import dataset_configs, model_configs
import exp_configs
from pyvideoai import config

import argparse
from pyvideoai.visualisations.metric_plotter import DefaultMetricPlotter
from pyvideoai.visualisations.telegram_reporter import DefaultTelegramReporter



from experiment_utils.argparse_utils import add_exp_arguments
def get_parser():
    parser = argparse.ArgumentParser(description="Plot stats to Telegram.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_exp_arguments(parser, 
            root_default=config.DEFAULT_EXPERIMENT_ROOT, dataset_default='hmdb', model_default='i3d_resnet50', name_default='crop224_8x8_largejit_plateau_1scrop5tcrop_split1',
            dataset_channel_choices=dataset_configs.available_channels, model_channel_choices=model_configs.available_channels, exp_channel_choices=exp_configs.available_channels)
    parser.add_argument("-l", "--load_epoch", type=int, default=None, help="Load from checkpoint. Set to -1 to load from the last checkpoint.")
    parser.add_argument("-B", "--telegram_bot_idx", type=int, default=0, help="Which Telegram bot to use defined in key.ini?")
    parser.add_argument("-v", "--version", type=str, default='auto', help="Experiment version (`auto` or integer). `auto` chooses the last version when resuming from the last, otherwise creates new version.")
    return parser





def main():
    parser = get_parser()
    args = parser.parse_args()

    cfg = exp_configs.load_cfg(args.dataset, args.model, args.experiment_name, args.dataset_channel, args.model_channel, args.experiment_channel)

    metrics = cfg.dataset_cfg.task.get_metrics(cfg)
    summary_fieldnames, summary_fieldtypes = ExperimentBuilder.return_fields_from_metrics(metrics)

    load_version = None
    if args.version == 'auto':
        _expversion = -2    # choose the last version
    else:
        _expversion = int(args.version)

    exp = ExperimentBuilder(args.experiment_root, args.dataset, args.model, args.experiment_name,
            summary_fieldnames = summary_fieldnames, summary_fieldtypes = summary_fieldtypes,
            version = _expversion, telegram_key_ini = config.KEY_INI_PATH, telegram_bot_idx = args.telegram_bot_idx)

    exp.load_summary()

    if args.load_epoch == -1 or args.load_epoch is None:
        load_epoch = int(load_exp.summary['epoch'].iloc[-1])
    elif args.load_epoch >= 0:
        load_epoch = args.load_epoch
    else:
        raise ValueError("Wrong args.load_epoch value: {:d}".format(args.load_epoch))

    start_epoch = load_epoch + 1

    # Crop summary. We don't need stats later than `start_epoch`
    exp.clip_summary_from_epoch(start_epoch)

    # Plotting metrics
    metric_plotter = cfg.metric_plotter if hasattr(cfg, 'metric_plotter') else DefaultMetricPlotter()
    metric_plotter.add_metrics(metrics)
    telegram_reporter = cfg.telegram_reporter if hasattr(cfg, 'telegram_reporter') else DefaultTelegramReporter()

    figs = metric_plotter.plot(exp)
    telegram_reporter.report(metrics, exp, figs)

if __name__ == '__main__':
    main()
