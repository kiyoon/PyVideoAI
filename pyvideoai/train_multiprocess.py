import numpy as np
import torch
import torch.distributed as dist
from torch import nn, optim

import pickle
import os
import sys
from shutil import copy2

import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from experiment_utils import ExperimentBuilder
import dataset_configs, model_configs
import exp_configs
from . import config

import argparse
import copy

from .utils import loader
from torch.utils.data.distributed import DistributedSampler
import time

from torch.optim.lr_scheduler import ReduceLROnPlateau

#import slowfast.utils.checkpoint as cu
from .utils import distributed as du
import json
from .utils import misc
from .utils.stdout_logger import OutputLogger

from .train_and_val import train_epoch, eval_epoch
from .train_and_val_multilabel import train_epoch as train_epoch_multilabel
from .train_and_val_multilabel import eval_epoch as eval_epoch_multilabel

from .metrics.accuracy import ClipAccuracyMetric, VideoAccuracyMetric

import coloredlogs, logging, verboselogs
logger = verboselogs.VerboseLogger(__name__)    # add logger.success


from ssl import SSLCertVerificationError
from urllib3.exceptions import NewConnectionError, MaxRetryError
from requests import ConnectionError

import configparser

import git

_SCRIPT_DIR = os.path.dirname(os.path.abspath( __file__ ))

def _suppress_print():
    """
    Suppresses printing from the current process.
    """

    sys.stdout = open(os.devnull,'w')
    sys.stderr = open(os.devnull,'w')


def distributed_init(global_seed, local_world_size):
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    local_rank = rank % local_world_size

    #local_seed = global_seed + rank
    local_seed = global_seed

    # Set GPU
    torch.cuda.set_device(local_rank)

    # Set seed
    torch.manual_seed(local_seed)
    torch.cuda.manual_seed(local_seed)
    np.random.seed(local_seed)
    # DALI seed

    return rank, world_size, local_rank, local_world_size, local_seed



def train(args):
    rank, world_size, local_rank, local_world_size, local_seed = distributed_init(args.seed, args.local_world_size)
    if rank == 0:
        coloredlogs.install(fmt='%(name)s: %(lineno)4d - %(levelname)s - %(message)s', level='INFO')
        logging.getLogger('slowfast.utils.checkpoint').setLevel(logging.WARNING)

    perform_multicropval = args.multi_crop_val_period > 0

    cfg = exp_configs.load_cfg(args.dataset, args.model, args.experiment_name, args.dataset_channel, args.model_channel, args.experiment_channel)

    if cfg.dataset_cfg.task == 'singlelabel_classification':
        summary_fieldnames, summary_fieldtypes = ExperimentBuilder.return_fields_singlelabel(multicropval = perform_multicropval)
        metrics = {'train': [ClipAccuracyMetric()],
                'val': [ClipAccuracyMetric()],
                'multicropval': [VideoAccuracyMetric(topk=(1,5))]
                }
    elif cfg.dataset_cfg.task == 'multilabel_classification':
        summary_fieldnames, summary_fieldtypes = ExperimentBuilder.return_fields_multilabel(multicropval = perform_multicropval)
    exp = ExperimentBuilder(args.experiment_root, args.dataset, args.model, args.experiment_name, summary_fieldnames = summary_fieldnames, summary_fieldtypes = summary_fieldtypes, telegram_key_ini = config.KEY_INI_PATH, telegram_bot_idx = args.telegram_bot_idx)

    if rank == 0:
        exp.make_dirs_for_training()

        f_handler = logging.FileHandler(os.path.join(exp.logs_dir, 'train.log'))
        #f_handler.setLevel(logging.NOTSET)
        f_handler.setLevel(logging.DEBUG)

        # Create formatters and add it to handlers
        f_format = logging.Formatter('%(asctime)s - %(name)s: %(lineno)4d - %(levelname)s - %(message)s')
        f_handler.setFormatter(f_format)

        # Add handlers to the logger
        root_logger = logging.getLogger()
        root_logger.addHandler(f_handler)
    else:
        _suppress_print()

    try:
        # Writes the pids to file, to make killing processes easier.    
        if world_size > 1:
            du.write_pids_to_file(os.path.join(config.PYVIDEOAI_DIR, 'tools', "last_pids.txt"))

        if rank == 0:
            repo = git.Repo(search_parent_directories=True)
            sha = repo.head.object.hexsha
            logger.info("git hash: %s", sha)

            # save configs
            exp.dump_args(args)
            logger.info("args: " + json.dumps(args.__dict__, sort_keys=False, indent=4))
            dataset_config_dir = os.path.join(exp.configs_dir, 'dataset_config')
            os.makedirs(dataset_config_dir, exist_ok=True)
            #copy2(os.path.join(dataset_configs._SCRIPT_DIR, '__init__.py'), os.path.join(dataset_config_dir, '__init__.py'))
            copy2(dataset_configs.config_path(args.dataset, args.dataset_channel), os.path.join(dataset_config_dir, args.dataset + '.py'))

            model_config_dir = os.path.join(exp.configs_dir, 'model_config')
            os.makedirs(model_config_dir, exist_ok=True)
            #copy2(os.path.join(model_configs._SCRIPT_DIR, '__init__.py'), os.path.join(model_config_dir, '__init__.py'))
            copy2(model_configs.config_path(args.model, args.model_channel), os.path.join(model_config_dir, args.model + '.py'))

            exp_config_dir = os.path.join(exp.configs_dir, 'exp_config')
            os.makedirs(exp_config_dir, exist_ok=True)
            config_file_name = exp_configs._get_file_name(args.dataset, args.model, args.experiment_name)
            #copy2(os.path.join(exp_configs._SCRIPT_DIR, '__init__.py'), os.path.join(config_dir, '__init__.py'))
            copy2(exp_configs.config_path(args.dataset, args.model, args.experiment_name, args.experiment_channel), os.path.join(exp_config_dir, os.path.basename(config_file_name)))


        # Dataset
        if perform_multicropval:
            splits = ['train', 'val', 'multicropval']
        else:
            splits = ['train', 'val']

        torch_datasets = {split: cfg.get_torch_dataset(split) for split in splits}
        data_unpack_funcs = {split: cfg.get_data_unpack_func(split) for split in splits}
        input_reshape_funcs= {split: cfg.get_input_reshape_func(split) for split in splits}
        batch_size = cfg.batch_size() if callable(cfg.batch_size) else cfg.batch_size
        logger.info(f'Using batch size of {batch_size} per process (per GPU), resulting in total size of {batch_size * world_size}.')

        train_sampler = DistributedSampler(torch_datasets['train']) if world_size > 1 else None
        val_sampler = DistributedSampler(torch_datasets['val'], shuffle=False) if world_size > 1 else None
        if perform_multicropval:
            multi_crop_val_sampler = DistributedSampler(torch_datasets['multicropval'], shuffle=False) if world_size > 1 else None
        train_dataloader = torch.utils.data.DataLoader(torch_datasets['train'], batch_size=batch_size, shuffle=False if train_sampler else True, sampler=train_sampler, num_workers=args.dataloader_num_workers, pin_memory=True, drop_last=True)
        val_dataloader = torch.utils.data.DataLoader(torch_datasets['val'], batch_size=batch_size, shuffle=False, sampler=val_sampler, num_workers=args.dataloader_num_workers, pin_memory=True, drop_last=False)
        if perform_multicropval:
            multi_crop_val_dataloader = torch.utils.data.DataLoader(torch_datasets['multicropval'], batch_size=batch_size, shuffle=False, sampler=multi_crop_val_sampler, num_workers=args.dataloader_num_workers, pin_memory=True, drop_last=False)
        # Use global seed because we have to maintain the consistency of shuffling between shards.
    #    train_dataloader = DALILoader(batch_size = 1, file_list = 'epic_verb_train_split.txt', uid2label = dataset_cfg.uid2label,
    #            sequence_length = 8, stride=16, crop_size= (224,224), num_threads=1, seed=args.seed, device_id=local_rank, shard_id=rank, num_shards=world_size, shuffle=True, pad_last_batch=True)
    #    val_dataloader = DALILoader(batch_size = 1, file_list = 'epic_verb_val_split.txt', uid2label = dataset_cfg.uid2label,
    #            sequence_length = 8, stride=16, crop_size= (224,224), test=True, num_threads=1, seed=args.seed, device_id=local_rank, shard_id=rank, num_shards=world_size, shuffle=False, pad_last_batch=False)

        # Network
        # Construct the model
        model = cfg.load_model()
        if hasattr(cfg, 'get_optim_policies'):
            policies = cfg.get_optim_policies(model)
        elif hasattr(cfg.model_cfg, 'get_optim_policies'):
            policies = cfg.model_cfg.get_optim_policies(model)
        else:
            policies = model.parameters()
        # Determine the GPU used by the current process
        cur_device = torch.cuda.current_device()
        # Transfer the model to the current GPU device
        model = model.to(device=cur_device, non_blocking=True)
        # Use multi-process data parallel model in the multi-gpu setting
        if world_size > 1:
            # Make model replica operate on the current device
            model = nn.parallel.DistributedDataParallel(
                module=model, device_ids=[cur_device], output_device=cur_device,
                find_unused_parameters=cfg.model_cfg.ddp_find_unused_parameters
            )

        misc.log_model_info(model)

        if args.load_epoch is not None:
            # Training and validation stats
            exp.load_summary()

            if args.load_epoch == -1:
                load_epoch = int(exp.summary['epoch'][-1])
            elif args.load_epoch >= 0:
                load_epoch = args.load_epoch
            else:
                raise ValueError("Wrong args.load_epoch value: {:d}".format(args.load_epoch))

            weights_path = exp.get_checkpoint_path(load_epoch)
            checkpoint = loader.model_load_weights_GPU(model, weights_path, cur_device, world_size)
            start_epoch = load_epoch + 1

        else:
            start_epoch = 0
            # Save training and validation stats
            if rank == 0:
                exp.init_summary()

            if hasattr(cfg, 'load_pretrained'):
                cfg.load_pretrained(model)



        if hasattr(cfg, 'criterion'):
            criterion = cfg.criterion()
        else:
            if cfg.dataset_cfg.task == 'singlelabel_classification':
                logger.info(f"cfg.criterion not defined. Using CrossEntropyLoss()")
                criterion = nn.CrossEntropyLoss()
            elif cfg.dataset_cfg.task == 'multilabel_classification':
                logger.info(f"cfg.criterion not defined. Using BCEWithLogitsLoss()")
                criterion = nn.BCEWithLogitsLoss()
            else:
                raise ValueError(f"cfg.dataset_cfg.task not known task: {cfg.dataset_cfg.task}")


        optimiser = cfg.optimiser(policies)
        if args.load_epoch is not None:
            try:
                # typically not done when fine-tunning.
                if hasattr(cfg, 'load_optimiser_state'):
                    load_optimiser_state = cfg.load_optimiser_state
                else:
                    logger.info("cfg.load_optimiser_state not found. By default, it will try to load.")
                    load_optimiser_state = True

                if load_optimiser_state:
                    logger.info("Loading optimiser state from the checkpoint.")
                    optimiser.load_state_dict(checkpoint["optimiser_state"])
            except ValueError as e:
                logger.error(f"Loading optimiser state_dict failed: {repr(e)}")
        iters_per_epoch = len(train_dataloader)

        scheduler = cfg.scheduler(optimiser, iters_per_epoch)
        if isinstance(scheduler, ReduceLROnPlateau):
            logger.info("ReduceLROnPlateau scheduler is selected. The `scheduler.step(val_loss)` function will be called at the end of epoch (after validation), but not every iteration.")
            if args.load_epoch is not None:
                logger.info("For the ReduceLROnPlateau scheduler, initialise using training stats (summary.csv) instead of the checkpoint.")
                for i in range(start_epoch):
                    with OutputLogger(scheduler.__module__, "INFO"):   # redirect stdout print() to logging (for verbose=True)
                        scheduler.step(exp.summary['val_loss'][i])
        else:
            logger.info("Scheduler is not ReduceLROnPlateau. The `scheduler.step()` function will be called every iteration (not every epoch).")
            if args.load_epoch is not None:
            #    for i in range(start_epoch*iters_per_epoch):
            #        scheduler.step()
                if scheduler is not None:
                    last_iter_num = start_epoch * iters_per_epoch 
                    if hasattr(cfg, 'load_scheduler_state'):
                        load_scheduler_state = cfg.load_scheduler_state
                    else:
                        logger.info("cfg.load_scheduler_state not found. By default, it will try to load.")
                        load_scheduler_state = True

                    if load_scheduler_state:
                        if checkpoint["scheduler_state"] is None:
                            logger.warning(f"Scheduler appears to have changed. Initialising using param last_epoch instead of using checkpoint scheduler state.")
                            scheduler = cfg.scheduler(optimiser, iters_per_epoch, last_epoch=last_iter_num-1)   # the last_epoch always has to be {the number of step() called} - 1
                        elif checkpoint["scheduler_state"]["last_epoch"] != last_iter_num:
                            logger.warning(f"Scheduler's last_epoch {checkpoint['scheduler_state']['last_epoch']} doesn't match with the epoch you're loading ({last_iter_num}). Possibly using different number of GPUs, batch size, or dataset size. Initialising using param last_epoch instead.")
                            scheduler = cfg.scheduler(optimiser, iters_per_epoch, last_epoch=last_iter_num-1)   # the last_epoch always has to be {the number of step() called} - 1
                        else:
                            try:
                                logger.info("Loading scheduler state from the checkpoint.")
                                scheduler.load_state_dict(checkpoint["scheduler_state"])
                            except Exception:
                                logger.warning(f"Scheduler appears to have changed. Initialising using param last_epoch instead of using checkpoint scheduler state.")
                                scheduler = cfg.scheduler(optimiser, iters_per_epoch, last_epoch=last_iter_num-1)   # the last_epoch always has to be {the number of step() called} - 1
                    else:
                        logger.info("Initialising scheduler using param last_epoch. NOT loading scheduler state from the checkpoint.")
                        scheduler = cfg.scheduler(optimiser, iters_per_epoch, last_epoch=last_iter_num-1)   # the last_epoch always has to be {the number of step() called} - 1



        #del checkpoint  # dereference seems crucial
        #torch.cuda.empty_cache()


        if rank == 0:
            train_writer = SummaryWriter(os.path.join(exp.tensorboard_runs_dir, 'train'), comment='train')
            val_writer = SummaryWriter(os.path.join(exp.tensorboard_runs_dir, 'val'), comment='val')
            if perform_multicropval:
                multi_crop_val_writer = SummaryWriter(os.path.join(exp.tensorboard_runs_dir, 'multi_crop_val'), comment='multi_crop_val')
            else:
                multi_crop_val_writer = None

            # to save maximum val_acc model, when option "args.save_mode" is "higher" or "last_and_peaks"
            max_val_metric = 0.
            max_val_epoch = -1
        else:
            train_writer = None
            val_writer = None
            multi_crop_val_writer = None

        for epoch in range(start_epoch, args.num_epochs):
            if hasattr(cfg, "epoch_start_script"):
                # structurise
                train_kit = {}
                train_kit["model"] = model
                train_kit["optimiser"] = optimiser 
                train_kit["scheduler"] = scheduler 
                train_kit["criterion"] = criterion 
                train_kit["train_dataloader"] = train_dataloader 
                train_kit["data_unpack_funcs"] = data_unpack_funcs 
                train_kit["input_reshape_funcs"] = input_reshape_funcs 
                # train_kit can be modified in the function
                cfg.epoch_start_script(epoch, copy.deepcopy(exp), copy.deepcopy(args), rank, world_size, train_kit)
                # unpack and apply the modifications
                model = train_kit["model"]
                optimiser = train_kit["optimiser"]
                scheduler = train_kit["scheduler"]
                criterion = train_kit["criterion"]
                train_dataloader = train_kit["train_dataloader"]
                data_unpack_funcs = train_kit["data_unpack_funcs"]
                input_reshape_funcs = train_kit["input_reshape_funcs"]

            # Shuffle the dataset.
            loader.shuffle_dataset(train_dataloader, epoch, args.seed)

            if rank == 0:
                print()
                logger.info("Epoch %d/%d" % (epoch, args.num_epochs-1))

            if cfg.dataset_cfg.task == 'singlelabel_classification':
                sample_seen, total_samples, loss, elapsed_time = train_epoch(model, optimiser, scheduler, criterion, train_dataloader, data_unpack_funcs['train'], metrics['train'], rank, world_size, input_reshape_func=input_reshape_funcs['train'])
                if rank == 0:#{{{
                    train_writer.add_scalar('Loss', loss, epoch)
                    train_writer.add_scalar('Accuracy', acc, epoch)
                    train_writer.add_scalar('Runtime_sec', elapsed_time, epoch)
                    train_writer.add_scalar('Sample_seen', sample_seen, epoch)
                    train_writer.add_scalar('Total_samples', total_samples, epoch)
                    # writer.add_graph(model, inputs)
                    curr_stat = {'epoch': epoch, 'train_runtime_sec': elapsed_time, 'train_loss': loss, 'train_acc': acc}#}}}
         
                val_sample_seen, val_total_samples, val_loss, val_elapsed_time, _ = eval_epoch(model, criterion, val_dataloader, data_unpack_funcs['val'], metrics['val'], cfg.dataset_cfg.num_classes, batch_size, True, rank, world_size, input_reshape_func=input_reshape_funcs['val'], scheduler=scheduler)
                val_metric = val_acc    # which metric to use for deciding the best model
                if rank == 0:#{{{
                    val_writer.add_scalar('Loss', val_loss, epoch)
                    val_writer.add_scalar('Accuracy', val_acc, epoch)
                    val_writer.add_scalar('Runtime_sec', val_elapsed_time, epoch)
                    val_writer.add_scalar('Sample_seen', val_sample_seen, epoch)
                    val_writer.add_scalar('Total_samples', val_total_samples, epoch)
                    curr_stat.update({'val_runtime_sec': val_elapsed_time, 'val_loss': val_loss, 'val_acc': val_acc})#}}}

                if perform_multicropval and epoch % args.multi_crop_val_period == args.multi_crop_val_period -1:
                    multi_crop_val_sample_seen, multi_crop_val_total_samples, multi_crop_val_loss, multi_crop_val_elapsed_time, _ = eval_epoch(model, criterion, multi_crop_val_dataloader, data_unpack_funcs['multicropval'], metrics['multicropval'], cfg.dataset_cfg.num_classes, batch_size, False, rank, world_size, input_reshape_func=input_reshape_funcs['multicropval'], scheduler=None)  # No scheduler needed for multicropval
                    if rank == 0:#{{{
                        multi_crop_val_writer.add_scalar('Loss', multi_crop_val_loss, epoch)
                        multi_crop_val_writer.add_scalar('Accuracy', multi_crop_val_acc, epoch)
                        multi_crop_val_writer.add_scalar('Video_accuracy_top-1', multi_crop_val_vid_acc_top1, epoch)
                        multi_crop_val_writer.add_scalar('Video_accuracy_top-5', multi_crop_val_vid_acc_top5, epoch)
                        multi_crop_val_writer.add_scalar('Runtime_sec', multi_crop_val_elapsed_time, epoch)
                        multi_crop_val_writer.add_scalar('Sample_seen', multi_crop_val_sample_seen, epoch)
                        multi_crop_val_writer.add_scalar('Total_samples', multi_crop_val_total_samples, epoch)
                        curr_stat.update({'multi_crop_val_runtime_sec': multi_crop_val_elapsed_time, 'multi_crop_val_loss': multi_crop_val_loss, 'multi_crop_val_acc': multi_crop_val_acc, 'multi_crop_val_vid_acc_top1': multi_crop_val_vid_acc_top1, 'multi_crop_val_vid_acc_top5': multi_crop_val_vid_acc_top5})#}}}

            elif cfg.dataset_cfg.task == 'multilabel_classification':
                sample_seen, total_samples, loss, vid_mAP, elapsed_time, _ = train_epoch_multilabel(model, optimiser, scheduler, criterion, train_dataloader, data_unpack_funcs['train'], rank, world_size, input_reshape_func=input_reshape_funcs['train'])
                if rank == 0:#{{{
                    train_writer.add_scalar('Loss', loss, epoch)
                    train_writer.add_scalar('mAP', vid_mAP, epoch)
                    train_writer.add_scalar('Runtime_sec', elapsed_time, epoch)
                    train_writer.add_scalar('Sample_seen', sample_seen, epoch)
                    train_writer.add_scalar('Total_samples', total_samples, epoch)
                    # writer.add_graph(model, inputs)
                    curr_stat = {'epoch': epoch, 'train_runtime_sec': elapsed_time, 'train_loss': loss, 'train_vid_mAP': vid_mAP}#}}}
         
                val_sample_seen, val_total_samples, val_loss, val_vid_mAP, val_elapsed_time, _, _ = eval_epoch_multilabel(model, criterion, val_dataloader, data_unpack_funcs['val'], cfg.dataset_cfg.num_classes, batch_size, True, rank, world_size, input_reshape_func=input_reshape_funcs['val'], scheduler=scheduler)
                val_metric = val_vid_mAP    # which metric to use for deciding the best model
                if rank == 0:#{{{
                    val_writer.add_scalar('Loss', val_loss, epoch)
                    val_writer.add_scalar('mAP', val_vid_mAP, epoch)
                    val_writer.add_scalar('Runtime_sec', val_elapsed_time, epoch)
                    val_writer.add_scalar('Sample_seen', val_sample_seen, epoch)
                    val_writer.add_scalar('Total_samples', val_total_samples, epoch)
                    curr_stat.update({'val_runtime_sec': val_elapsed_time, 'val_loss': val_loss, 'val_vid_mAP': val_vid_mAP})#}}}

                if perform_multicropval and epoch % args.multi_crop_val_period == args.multi_crop_val_period -1:
                    multi_crop_val_sample_seen, multi_crop_val_total_samples, multi_crop_val_loss, multi_crop_val_vid_mAP, multi_crop_val_elapsed_time, _, _ = eval_epoch_multilabel(model, criterion, multi_crop_val_dataloader, data_unpack_funcs['multicropval'], cfg.dataset_cfg.num_classes, batch_size, False, rank, world_size, input_reshape_func=input_reshape_funcs['multicropval'], scheduler=None)  # No scheduler needed for multicropval
                    if rank == 0:#{{{
                        multi_crop_val_writer.add_scalar('Loss', multi_crop_val_loss, epoch)
                        multi_crop_val_writer.add_scalar('mAP', multi_crop_val_vid_mAP, epoch)
                        multi_crop_val_writer.add_scalar('Runtime_sec', multi_crop_val_elapsed_time, epoch)
                        multi_crop_val_writer.add_scalar('Sample_seen', multi_crop_val_sample_seen, epoch)
                        multi_crop_val_writer.add_scalar('Total_samples', multi_crop_val_total_samples, epoch)
                        curr_stat.update({'multi_crop_val_runtime_sec': multi_crop_val_elapsed_time, 'multi_crop_val_loss': multi_crop_val_loss, 'multi_crop_val_vid_mAP': multi_crop_val_vid_mAP})#}}}
            else:
                raise ValueError(f"Unknown task: {cfg.dataset_cfg.task}")

            early_stopping = False      # need it for the entire process, because later we'll broadcast

            if rank == 0:
                exp.add_summary_line(curr_stat)

                if hasattr(cfg, 'early_stopping_condition'):
                    early_stopping = cfg.early_stopping_condition(epoch, exp)

                send_telegram = early_stopping or (epoch % args.telegram_post_period == args.telegram_post_period -1)
                try:
                    start_time_plot = time.time()
                    exp.plot_summary(send_telegram = send_telegram)
                except (SSLCertVerificationError, OSError, NewConnectionError, MaxRetryError, ConnectionError) as e:
                    """Usually, max-retries exceeds when you run it for a while.
                    Therefore, even if it fails to send the plots, don't crash the programme and keep it running.
                    """
                    logger.exception('Failed to send plots to Telegram.')

                elapsed_time_plot = time.time() - start_time_plot

                if args.telegram_post_period < 100 and send_telegram and elapsed_time_plot > 5 * 60:
                    # Sending plots to telegram took longer than 5 mins.
                    # Telegram is limiting the traffic due to heavy usage.
                    logger.warning('Sending plots to Telegram took over 5 mins. Setting the plotting period to 100 epochs.')
                    args.telegram_post_period = 100


                is_higher = val_metric >= max_val_metric - 1e-6
                if is_higher:
                    max_val_metric = val_metric
                    max_val_epoch = epoch

                if is_higher or args.save_mode in ["all", "last_and_peaks"]:
                    # all, last_and_peaks: save always
                    # higher: save model when higher
                    model_path = exp.get_checkpoint_path(epoch) 
                    io_error = True 
                    while io_error:
                        try:
                            logger.info(f"Saving model to {model_path}")
                            state_dict = model.module.state_dict() if world_size > 1 else model.state_dict()
                            checkpoint = {
                                    "epoch": epoch,
                                    "model_state": state_dict,
                                    "optimiser_state": optimiser.state_dict(),
                                    "scheduler_state": None if scheduler is None else scheduler.state_dict()
                                }
                            torch.save(checkpoint, model_path)
                            io_error = False
                        except IOError as e:
                            logger.exception("IOError whilst saving the model.")
                            input("Press Enter to retry: ")

                if args.save_mode == "last_and_peaks":
                    if epoch > 0 and epoch - 1 != max_val_epoch:
                        # delete previous checkpoint if not best
                        model_path = exp.get_checkpoint_path(epoch-1) 
                        io_error = True 
                        while io_error:
                            try:
                                logger.info(f"Removing previous model: {model_path}")
                                os.remove(model_path)
                                io_error = False
                            except IOError as e:
                                logger.exception("IOError whilst removing the model.")
                                input("Press Enter to retry: ")
                
            if hasattr(cfg, 'early_stopping_condition'):
                if world_size > 1:
                    # Broadcast the early stopping flag to the entire process.
                    early_stopping_flag = torch.ByteTensor([early_stopping]).to(cur_device)
                    dist.broadcast(early_stopping_flag, 0)
                    # Copy from GPU to CPU (sync point)
                    early_stopping_flag = bool(early_stopping_flag.item())
                else:
                    early_stopping_flag = early_stopping

                if early_stopping_flag:
                    logger.info("Early stopping triggered.")
                    break


        logger.success('Finished training')
        if rank == 0:
            exp.tg_send_text_with_expname('Finished training')

    except Exception as e:
        logger.exception("Exception occurred")
        # Every process is going to send exception.
        # This can make your Telegram report filled with many duplicates,
        # but at the same time it ensures that you receive a message when anything wrong happens.
#        if rank == 0:
        exp.tg_send_text_with_expname('Exception occurred\n\n' + repr(e))


