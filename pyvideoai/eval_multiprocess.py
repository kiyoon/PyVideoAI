import numpy as np
import random
import torch
import torch.distributed as dist
from torch import nn, optim

import pickle
import os
import sys
from shutil import copy2

from experiment_utils import ExperimentBuilder
import dataset_configs, model_configs, exp_configs
from . import config

import json
import argparse

from .utils import loader
from torch.utils.data.distributed import DistributedSampler

from .utils import distributed as du
from .utils import misc
from .train_and_eval import eval_epoch

from .metrics.metric import ClipPredictionsGatherer, VideoPredictionsGatherer

import coloredlogs, logging, verboselogs
logger = verboselogs.VerboseLogger(__name__)    # add logger.success


import configparser

# Version checking
from . import __version__
import socket


_SCRIPT_DIR = os.path.dirname(os.path.abspath( __file__ ))


def evaluation(args):
    rank, world_size, local_rank, local_world_size, local_seed = du.distributed_init(args.seed, args.local_world_size)
    if rank == 0:
        coloredlogs.install(fmt='%(name)s: %(lineno)4d - %(levelname)s - %(message)s', level='INFO')
        #logging.getLogger('pyvideoai.slowfast.utils.checkpoint').setLevel(logging.WARNING)

    cfg = exp_configs.load_cfg(args.dataset, args.model, args.experiment_name, args.dataset_channel, args.model_channel, args.experiment_channel)

    metrics = cfg.dataset_cfg.task.get_metrics(cfg)

    if args.version == 'auto':
        _expversion = -2    # last version (do not create new)
    else:
        _expversion = int(args.version)

    summary_fieldnames, summary_fieldtypes = ExperimentBuilder.return_fields_from_metrics(metrics)
    exp = ExperimentBuilder(args.experiment_root, args.dataset, args.model, args.experiment_name, summary_fieldnames = summary_fieldnames, summary_fieldtypes = summary_fieldtypes, version = _expversion, telegram_key_ini = config.KEY_INI_PATH, telegram_bot_idx = args.telegram_bot_idx)



    if rank == 0:
        exp.make_dirs_for_training()

        f_handler = logging.FileHandler(os.path.join(exp.logs_dir, 'val.log'))
        #f_handler.setLevel(logging.NOTSET)
        f_handler.setLevel(logging.DEBUG)

        # Create formatters and add it to handlers
        f_format = logging.Formatter('%(asctime)s - %(name)s: %(lineno)4d - %(levelname)s - %(message)s')
        f_handler.setFormatter(f_format)

        # Add handlers to the logger
        root_logger = logging.getLogger()
        root_logger.addHandler(f_handler)
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
            dataset_config_dir = os.path.join(exp.configs_dir, 'dataset_config_val')
            os.makedirs(dataset_config_dir, exist_ok=True)
            #copy2(os.path.join(dataset_configs._SCRIPT_DIR, '__init__.py'), os.path.join(dataset_config_dir, '__init__.py'))
            copy2(dataset_configs.config_path(args.dataset, args.dataset_channel), os.path.join(dataset_config_dir, args.dataset + '.py'))

            model_config_dir = os.path.join(exp.configs_dir, 'model_config_val')
            os.makedirs(model_config_dir, exist_ok=True)
            #copy2(os.path.join(model_configs._SCRIPT_DIR, 'model_configs', '__init__.py'), os.path.join(model_config_dir, '__init__.py'))
            copy2(model_configs.config_path(args.model, args.model_channel), os.path.join(model_config_dir, args.model + '.py'))

            exp_config_dir = os.path.join(exp.configs_dir, 'exp_config_val')
            os.makedirs(exp_config_dir, exist_ok=True)
            config_file_name = exp_configs._get_file_name(args.dataset, args.model, args.experiment_name)
            #copy2(os.path.join(exp_configs._SCRIPT_DIR, '__init__.py'), os.path.join(config_dir, '__init__.py'))
            copy2(exp_configs.config_path(args.dataset, args.model, args.experiment_name, args.experiment_channel), os.path.join(exp_config_dir, os.path.basename(config_file_name)))


        # Dataset
        if args.split is not None:
            split = args.split
        else:   # set split automatically
            if args.mode == 'oneclip':
                split = 'val'
            else:   # multicrop
                split = 'multicropval'

        if args.save_predictions:
            predictions_gatherer = cfg.dataset_cfg.task.get_predictions_gatherers(cfg)[split]
            metrics[split].append(predictions_gatherer)

        val_dataset = cfg.get_torch_dataset(split)
        data_unpack_func = cfg.get_data_unpack_func(split)


        input_reshape_func = cfg.get_input_reshape_func(split)
        batch_size = cfg.batch_size() if callable(cfg.batch_size) else cfg.batch_size
        logger.info(f'Using batch size of {batch_size} per process (per GPU), resulting in total size of {batch_size * world_size}.')

        val_sampler = DistributedSampler(val_dataset, shuffle=False) if world_size > 1 else None
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, sampler=val_sampler, num_workers=args.dataloader_num_workers, pin_memory=True, drop_last=False, worker_init_fn = du.seed_worker)

        # Network
        # Construct the model
        model = cfg.load_model()
        # Determine the GPU used by the current process
        cur_device = torch.cuda.current_device()
        # Transfer the model to the current GPU device
        model = model.to(device=cur_device, non_blocking=True)
        # Use multi-process data parallel model in the multi-gpu setting
        if world_size > 1:
            # Make model replica operate on the current device
            model = nn.parallel.DistributedDataParallel(
                module=model, device_ids=[cur_device], output_device=cur_device
            )

        misc.log_model_info(model)

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
            checkpoint = loader.model_load_weights_GPU(model, weights_path)

            logger.info("Training stats: %s", json.dumps(exp.get_epoch_stat(load_epoch), sort_keys=False, indent=4))
        else:
            if hasattr(cfg, 'load_pretrained'):
                cfg.load_pretrained(model)


        criterion = cfg.dataset_cfg.task.get_criterion(cfg)

        oneclip = args.mode == 'oneclip'

        if rank == 0:
            exp.tg_send_text_with_expname(f'Starting evaluation..')

        _, _, loss, elapsed_time, eval_log_str = eval_epoch(model, criterion, val_dataloader, data_unpack_func, metrics[split], None, cfg.dataset_cfg.num_classes, oneclip, rank, world_size, input_reshape_func=input_reshape_func, refresh_period=args.refresh_period)

        if rank == 0:
            # Update summary.csv
            curr_stat = {'epoch': load_epoch, f'{split}_runtime_sec': elapsed_time, f'{split}_loss': loss}

            for metric in metrics[split]:
                csv_fieldnames = metric.get_csv_fieldnames()
                if not isinstance(csv_fieldnames, (list, tuple)):
                    csv_fieldnames = (csv_fieldnames,)
                last_calculated_metrics = metric.last_calculated_metrics
                if not isinstance(last_calculated_metrics, (list, tuple)):
                    last_calculated_metrics = (last_calculated_metrics,)

                for csv_fieldname, last_calculated_metric in zip(csv_fieldnames, last_calculated_metrics):
                    if csv_fieldname is not None:   # PredictionsGatherers return None for the CSV fieldname.
                        curr_stat[csv_fieldname] = last_calculated_metric

            if load_epoch is not None and load_epoch >= 0:
                logger.info(f'Updating exp--summary.csv line with {curr_stat}.')
                exp.update_summary_line(curr_stat)
            else:
                logger.info(f'epoch {load_epoch} is not supported. Not updating the summary.csv line.')

            # save predictions
            if args.save_predictions:
                video_predictions, video_labels, video_ids = predictions_gatherer.get_predictions_numpy()

                if load_epoch is None:
                    predictions_file_path = os.path.join(exp.predictions_dir, 'pretrained_%sval.pkl' % (args.mode))
                else:
                    predictions_file_path = os.path.join(exp.predictions_dir, 'epoch_%04d_%sval.pkl' % (load_epoch, args.mode))
                os.makedirs(exp.predictions_dir, exist_ok=True)
                
                print("Saving predictions to: " + predictions_file_path) 
                with open(predictions_file_path, 'wb') as f:
                    pickle.dump({'video_predictions': video_predictions, 'video_labels': video_labels, 'video_ids': video_ids}, f, pickle.HIGHEST_PROTOCOL)



        logger.success('Finished evaluation')
        if rank == 0:
            exp.tg_send_text_with_expname('Finished evaluation\n\n' + eval_log_str)

    except Exception as e:
        logger.exception("Exception occurred")
        if rank == 0:
            exp.tg_send_text_with_expname('Exception occurred\n\n' + repr(e))

