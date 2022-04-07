import torch
from torch import nn

import pickle
import os

from experiment_utils import ExperimentBuilder
import dataset_configs, model_configs, exp_configs
from . import config

import json

from torch.utils.data.distributed import DistributedSampler

from .utils import distributed as du
from .utils import misc
from .train_and_eval import extract_features

# Version checking
from . import __version__
import socket

import traceback

import coloredlogs, logging, verboselogs
logger = verboselogs.VerboseLogger(__name__)    # add logger.success

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def feature_extraction(args):
    if hasattr(args, 'local_world_size'):
        local_world_size = args.local_world_size
    else:
        local_world_size = None
    rank, world_size, local_rank, local_world_size, local_seed = du.distributed_init(args.seed, local_world_size)
    if rank == 0:
        coloredlogs.install(fmt='', level=logging.NOTSET, stream=open(os.devnull, 'w'))  # Will set the output stream later on with custom level.

    cfg = exp_configs.load_cfg(args.dataset, args.model, args.experiment_name, args.dataset_channel, args.model_channel, args.experiment_channel)

    metrics = cfg.dataset_cfg.task.get_metrics(cfg)

    if args.version == 'auto':
        if args.load_epoch is not None:
            _expversion = 'last'
        else:
            _expversion = 'new'
    else:
        _expversion = int(args.version)

    summary_fieldnames, summary_fieldtypes = ExperimentBuilder.return_fields_from_metrics(metrics)
    exp = ExperimentBuilder(args.experiment_root, args.dataset, args.model, args.experiment_name, args.subfolder_name, summary_fieldnames = summary_fieldnames, summary_fieldtypes = summary_fieldtypes, version = _expversion, telegram_key_ini = config.KEY_INI_PATH, telegram_bot_idx = args.telegram_bot_idx)



    if rank == 0:
        exp.make_dirs_for_training()

        misc.install_colour_logger(level=args.console_log_level)
        misc.install_file_loggers(exp.logs_dir, file_prefix='extract_features')
    else:
        du.suppress_print()

    try:
        # Writes the pids to file, to make killing processes easier.
        if world_size > 1:
            du.write_pids_to_file(os.path.join(config.PYVIDEOAI_DIR, 'tools', "last_pids.txt"))

        if rank == 0:
            logger.info(f"PyTorch=={torch.__version__}")
            logger.info(f"PyVideoAI=={__version__}")
            logger.info(f"Experiment folder: {exp.experiment_dir} on host {socket.gethostname()}")

            # save configs
#            exp.dump_args(args)
            logger.info("args: " + json.dumps(args.__dict__, sort_keys=False, indent=4))
            du.log_distributed_info(world_size, local_world_size)
            dataset_config_dir = os.path.join(exp.configs_dir, 'dataset_configs_extract_features')
            dataset_configs.copy_cfg_files(cfg.dataset_cfg, dataset_config_dir)

            model_config_dir = os.path.join(exp.configs_dir, 'model_configs_extract_features')
            model_configs.copy_cfg_files(cfg.model_cfg, model_config_dir)

            exp_config_dir = os.path.join(exp.configs_dir, 'exp_configs_extract_features')
            exp_configs.copy_cfg_files(cfg, exp_config_dir)

        # Dataset
        if args.split is not None:
            split = args.split
        else:   # set split automatically
            if args.mode == 'oneclip':
                split = 'val'
            else:   # multicrop
                split = 'multicropval'


        val_dataset = cfg.get_torch_dataset(split)
        data_unpack_func = cfg.get_data_unpack_func(split)


        input_reshape_func = cfg.get_input_reshape_func(split)
        if hasattr(cfg, 'val_batch_size'):
            batch_size = cfg.val_batch_size() if callable(cfg.val_batch_size) else cfg.val_batch_size
        else:
            batch_size = cfg.batch_size() if callable(cfg.batch_size) else cfg.batch_size
        logger.info(f'Using batch size of {batch_size} per process (per GPU), resulting in total size of {batch_size * world_size}.')

        val_sampler = DistributedSampler(val_dataset, shuffle=False) if world_size > 1 else None
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, sampler=val_sampler, num_workers=args.dataloader_num_workers, pin_memory=True, drop_last=False, worker_init_fn = du.seed_worker)

        # Network
        # Construct the model
        model = cfg.load_model()
        # Determine the GPU used by the current process
        cur_device = torch.cuda.current_device()

        # Load weights
        if args.load_epoch == -1:
            exp.load_summary()
            load_epoch = int(exp.summary['epoch'].iloc[-1])
        elif args.load_epoch == -2:
            exp.load_summary()
            best_metric, best_metric_fieldname = metrics.get_best_metric_and_fieldname()
            logger.info(f'Using the best metric from CSV field `{best_metric_fieldname}`')
            load_epoch = int(exp.get_best_model_stat(best_metric_fieldname, best_metric.is_better)['epoch'])
        elif args.load_epoch is None:
            load_epoch = None
        elif args.load_epoch >= 0:
            exp.load_summary()
            load_epoch = args.load_epoch
        else:
            raise ValueError(f"Wrong args.load_epoch value: {args.load_epoch:d}")

        if load_epoch is not None:
            weights_path = exp.get_checkpoint_path(load_epoch)
            checkpoint = torch.load(weights_path, map_location=f'cuda:{cur_device}')
            model.load_state_dict(checkpoint['model_state'])

            logger.info("Training stats: %s", json.dumps(exp.get_epoch_stat(load_epoch), sort_keys=False, indent=4))
        else:
            if hasattr(cfg, 'load_pretrained'):
                cfg.load_pretrained(model)

        # Convert the model to feature model
        if hasattr(cfg, 'feature_extract_model'):
            model = cfg.feature_extract_model(model, args.featuremodel_name)
        else:
            model = cfg.model_cfg.feature_extract_model(model, args.featuremodel_name)

        # Transfer the model to the current GPU device
        model = model.to(device=cur_device, non_blocking=True)
        # Use multi-process data parallel model in the multi-gpu setting
        if world_size > 1:
            # Make model replica operate on the current device
            model = nn.parallel.DistributedDataParallel(
                module=model, device_ids=[cur_device], output_device=cur_device
            )

        misc.log_model_info(model)
        misc.check_pillow_performance()


        if rank == 0:
            exp.tg_send_text_with_expname('Starting to extract features..')

        feature_data, _, _, _, eval_log_str = extract_features(model, val_dataloader, data_unpack_func, cfg.dataset_cfg.num_classes, split, rank, world_size, input_reshape_func=input_reshape_func, refresh_period=args.refresh_period)

        if rank == 0:
            # save features
            if load_epoch is None:
                features_file_path = os.path.join(exp.predictions_dir, f'{args.featuremodel_name}_pretrained_{split}_{args.mode}.pkl')
            else:
                features_file_path = os.path.join(exp.predictions_dir, f'{args.featuremodel_name}_epoch_{load_epoch:04d}_{split}_{args.mode}.pkl')
                os.makedirs(exp.predictions_dir, exist_ok=True)

                print("Saving features to: " + features_file_path)
                with open(features_file_path, 'wb') as f:
                    pickle.dump(feature_data, f, pickle.HIGHEST_PROTOCOL)


        logger.success('Finished extracting features')
        if rank == 0:
            exp.tg_send_text_with_expname('Finished extracting features\n\n' + eval_log_str)

    except Exception:
        logger.exception("Exception occurred whilst extracting features")
        # Every process is going to send exception.
        # This can make your Telegram report filled with many duplicates,
        # but at the same time it ensures that you receive a message when anything wrong happens.
#        if rank == 0:
        exp.tg_send_text_with_expname('Exception occurred whilst extracting features\n\n' + traceback.format_exc())
