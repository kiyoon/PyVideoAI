import torch
import torch.distributed as dist
from torch import nn

import os

import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from experiment_utils import ExperimentBuilder
import dataset_configs, model_configs
import exp_configs
from . import config

import copy

from .utils import loader
from torch.utils.data.distributed import DistributedSampler
import time


from .utils import distributed as du
import json
import yaml
from yaml.loader import SafeLoader
from .utils import misc
from .utils.stdout_logger import OutputLogger
from .utils.distributed import check_random_seed_in_sync

from .train_and_eval import train_epoch, eval_epoch, get_scheduler_type

from .visualisations.metric_plotter import DefaultMetricPlotter
from .visualisations.telegram_reporter import DefaultTelegramReporter

from ssl import SSLCertVerificationError
from urllib3.exceptions import NewConnectionError, MaxRetryError
from requests import ConnectionError

# Version checking
from . import __version__
import socket

import traceback
import coloredlogs, logging, verboselogs
logger = verboselogs.VerboseLogger(__name__)    # add logger.success

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))





def train(args):
    if hasattr(args, 'local_world_size'):
        local_world_size = args.local_world_size
    else:
        local_world_size = None
    rank, world_size, local_rank, local_world_size, local_seed = du.distributed_init(args.seed, local_world_size)
    if rank == 0:
        coloredlogs.install(fmt='', level=logging.NOTSET, stream=open(os.devnull, 'w'))  # Will set the output stream later on with custom level.

    perform_multicropval = args.multi_crop_val_period > 0

    cfg = exp_configs.load_cfg(args.dataset, args.model, args.experiment_name, args.dataset_channel, args.model_channel, args.experiment_channel)

    metrics = cfg.dataset_cfg.task.get_metrics(cfg)
    summary_fieldnames, summary_fieldtypes = ExperimentBuilder.return_fields_from_metrics(metrics)

    load_version = None
    if args.version == 'auto':
        if args.load_epoch == -1:
            _expversion = 'last'
        elif args.load_epoch is None:
            _expversion = 'new'
        else:
            # Load from intermediate (not last) checkpoint.
            # Copy stats from the last version (until the loading epoch) and continue with a new version.
            # Thus, load_version and _expversion is different
            load_version = 'last'
            _expversion = 'new'

    else:
        if args.load_epoch == -1 or args.load_epoch is None:
            _expversion = int(args.version)
        else:
            load_version = int(args.version)
            _expversion = 'new'

    exp = ExperimentBuilder(args.experiment_root, args.dataset, args.model, args.experiment_name, args.subfolder_name,
            summary_fieldnames = summary_fieldnames, summary_fieldtypes = summary_fieldtypes,
            version = _expversion, telegram_key_ini = config.KEY_INI_PATH, telegram_bot_idx = args.telegram_bot_idx)
    du.synchronize()  # before creating any experiment directory, we need to make sure that all versions for all processes is equal.
    if rank == 0:
        exp._check_old_structure_and_move()

    if load_version is None:
        load_exp = exp
    else:
        load_exp = ExperimentBuilder(args.experiment_root, args.dataset, args.model, args.experiment_name, args.subfolder_name,
                summary_fieldnames = summary_fieldnames, summary_fieldtypes = summary_fieldtypes,
                version = load_version, telegram_key_ini = config.KEY_INI_PATH, telegram_bot_idx = args.telegram_bot_idx)
        du.synchronize()  # before creating any experiment directory, we need to make sure that all versions for all processes is equal.

        load_exp.load_summary()
        exp.copy_from(load_exp, copy_dirs = rank == 0, exclude_weights=True)

    if rank == 0:
        exp.make_dirs_for_training()

        misc.install_colour_logger(level=args.console_log_level)
        misc.install_file_loggers(exp.logs_dir, file_prefix='train')
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

            # Enable Weights & Biases visualisation service.
            if args.wandb_project is not None:
                with OutputLogger('wandb', input='stderr'):
                    import wandb
                    wandb_rootdir = exp.experiment_dir
                    wandb_config = args.__dict__

                    if args.wandb_run_id is None:
                        wandb_dir = os.path.join(exp.experiment_dir, 'wandb')
                        if os.path.isdir(wandb_dir):
                            # If this directory exists before initialising, it means you're resuming the experiment.
                            wandb_run_ids = [filename.split('-')[-1] for filename in os.listdir(wandb_dir) if filename.startswith("run-")]
                            assert len(set(wandb_run_ids)) == 1, f'No run or more than one W&B runs detected in a single experiment folder: {wandb_dir}'
                            wandb_run_id = wandb_run_ids[0]
                            logger.info(f'You are resuming the experiment. We will resume W&B on run ID: {wandb_run_id} as well.')
                            # Detect project name automatically.
                            wandb_run_dirs = sorted([filename for filename in os.listdir(wandb_dir) if filename.startswith("run-")])
                            wandb_run_dir = wandb_run_dirs[-1]
                            wandb_run_config_path = os.path.join(wandb_dir, wandb_run_dir, 'files', 'config.yaml')
                            with open(wandb_run_config_path, 'r') as f:
                                wandb_run_config = yaml.load(f, Loader=SafeLoader)
                            wandb_run_project = wandb_run_config['wandb_project']['value']

                            if wandb_run_project != args.wandb_project:
                                logger.warning(f'W&B project name for the previous run ({wandb_run_project}) is different from {args.wandb_project = }. args will be first tried and if fails we will use the previous name.')
                                try:
                                    wandb.init(dir=wandb_rootdir, config=wandb_config, entity=args.wandb_entity, project=args.wandb_project, id=wandb_run_id, resume='must')
                                except wandb.errors.UsageError:
                                    logger.error(f'W&B project name {args.wandb_project = } did not work. Trying {wandb_run_project}.')
                                    wandb.init(dir=wandb_rootdir, config=wandb_config, entity=args.wandb_entity, project=wandb_run_project, id=wandb_run_id, resume='must')
                            else:
                                wandb.init(dir=wandb_rootdir, config=wandb_config, entity=args.wandb_entity, project=args.wandb_project, id=wandb_run_id, resume='must')
                        else:
                            wandb.init(dir=wandb_rootdir, config=wandb_config, entity=args.wandb_entity, project=args.wandb_project,
                                name=exp.full_exp_name)
                    else:
                        wandb.init(dir=wandb_rootdir, config=wandb_config, entity=args.wandb_entity, project=args.wandb_project, id=args.wandb_run_id, resume='must')
                    logger.info((f'Weights & Biases initialised.\n'
                        f'View project at {wandb.run.get_project_url()}\n'
                        f'View run at {wandb.run.get_url()}'))

                    wandb_details = {}
                    wandb_details['entity'] = wandb.run.entity
                    wandb_details['run_id'] = wandb.run.id
                    wandb_details['project'] = wandb.run.project
                    wandb_details['name'] = wandb.run.name
                    wandb_details['project_url'] = wandb.run.get_project_url()
                    wandb_details['run_url'] = wandb.run.get_url()

                    with open(os.path.join(exp.logs_dir, 'wandb.json'), 'w') as f:
                        json.dump(wandb_details, f, sort_keys=False, indent=4)


            # save configs
            exp.dump_args(args)
            logger.info("args: " + json.dumps(args.__dict__, sort_keys=False, indent=4))
            du.log_distributed_info(world_size, local_world_size)
            dataset_config_dir = os.path.join(exp.configs_dir, 'dataset_configs')
            dataset_configs.copy_cfg_files(cfg.dataset_cfg, dataset_config_dir)

            model_config_dir = os.path.join(exp.configs_dir, 'model_configs')
            model_configs.copy_cfg_files(cfg.model_cfg, model_config_dir)

            exp_config_dir = os.path.join(exp.configs_dir, 'exp_configs')
            exp_configs.copy_cfg_files(cfg, exp_config_dir)

            if args.wandb_project is not None:
                dataset_configs.wandb_log_cfg_files(cfg.dataset_cfg)
                model_configs.wandb_log_cfg_files(cfg.model_cfg)
                exp_configs.wandb_log_cfg_files(cfg)


        # Dataset
        if perform_multicropval:
            splits = ['train', 'val', 'multicropval']
        else:
            splits = ['train', 'val']

        torch_datasets = {split: cfg.get_torch_dataset(split) for split in splits}
        data_unpack_funcs = {split: cfg.get_data_unpack_func(split) for split in splits}
        input_reshape_funcs = {split: cfg.get_input_reshape_func(split) for split in splits}
        batch_size = cfg.batch_size() if callable(cfg.batch_size) else cfg.batch_size
        if hasattr(cfg, 'val_batch_size'):
            val_batch_size = cfg.val_batch_size() if callable(cfg.val_batch_size) else cfg.val_batch_size
        else:
            val_batch_size = batch_size
        logger.info(f'Using batch size of {batch_size} per process (per GPU), resulting in total size of {batch_size * world_size}.')
        logger.info(f'Using validation batch size of {val_batch_size} per process (per GPU), resulting in total size of {val_batch_size * world_size}.')


        train_sampler = DistributedSampler(torch_datasets['train']) if world_size > 1 else None
        val_sampler = DistributedSampler(torch_datasets['val'], shuffle=False) if world_size > 1 else None
        if perform_multicropval:
            multi_crop_val_sampler = DistributedSampler(torch_datasets['multicropval'], shuffle=False) if world_size > 1 else None
        train_dataloader = torch.utils.data.DataLoader(torch_datasets['train'], batch_size=batch_size, shuffle=False if train_sampler else True, sampler=train_sampler, num_workers=args.dataloader_num_workers, pin_memory=True, drop_last=True, worker_init_fn = du.seed_worker)
        val_dataloader = torch.utils.data.DataLoader(torch_datasets['val'], batch_size=val_batch_size, shuffle=False, sampler=val_sampler, num_workers=args.dataloader_num_workers, pin_memory=True, drop_last=False, worker_init_fn = du.seed_worker)
        if perform_multicropval:
            multi_crop_val_dataloader = torch.utils.data.DataLoader(torch_datasets['multicropval'], batch_size=val_batch_size, shuffle=False, sampler=multi_crop_val_sampler, num_workers=args.dataloader_num_workers, pin_memory=True, drop_last=False, worker_init_fn = du.seed_worker)
        # Use global seed because we have to maintain the consistency of shuffling between shards.
    #    train_dataloader = DALILoader(batch_size = 1, file_list = 'epic_verb_train_split.txt', uid2label = dataset_cfg.uid2label,
    #            sequence_length = 8, stride=16, crop_size= (224,224), num_threads=1, seed=args.seed, device_id=local_rank, shard_id=rank, num_shards=world_size, shuffle=True, pad_last_batch=True)
    #    val_dataloader = DALILoader(batch_size = 1, file_list = 'epic_verb_val_split.txt', uid2label = dataset_cfg.uid2label,
    #            sequence_length = 8, stride=16, crop_size= (224,224), test=True, num_threads=1, seed=args.seed, device_id=local_rank, shard_id=rank, num_shards=world_size, shuffle=False, pad_last_batch=False)


        def search_expcfg_then_modelcfg(name, default):
            if hasattr(cfg, name):
                return getattr(cfg, name)
            elif hasattr(cfg.model_cfg, name):
                return getattr(cfg.model_cfg, name)
            return default

        # check it before and after loading the model.
        if world_size > 1:
            logger.info('Checking if random seeds are in sync across processes, before and after loading the model. This ensures that the model is initialised with the same weights across processes.')
            check_random_seed_in_sync()

        # Network
        # Construct the model
        model = cfg.load_model()
        if hasattr(cfg, 'get_optim_policies'):
            policies = cfg.get_optim_policies(model)
        elif hasattr(cfg.model_cfg, 'get_optim_policies'):
            policies = cfg.model_cfg.get_optim_policies(model)
        else:
            policies = model.parameters()

        if hasattr(cfg, 'load_pretrained'):
            logger.warning('The behaviour of load_pretrained has changed. Now it is called before DDP, and the model is on CPU. Use `load_pretrained_postDDP` if your loading script expects the model to be DDP')
            cfg.load_pretrained(model)

        # Determine the GPU used by the current process
        cur_device = torch.cuda.current_device()
        # Transfer the model to the current GPU device
        model = model.to(device=cur_device, non_blocking=True)
        # Use multi-process data parallel model in the multi-gpu setting
        if world_size > 1:
            ddp_find_unused_parameters = search_expcfg_then_modelcfg('ddp_find_unused_parameters', True)
            if ddp_find_unused_parameters:
                logger.info('Will find unused parameters for distributed training. This introduces extra overheads, so set ddp_find_unused_parameters to True only when necessary.')
            else:
                logger.info('Will NOT find unused parameters for distributed training. If you see an error, consider setting ddp_find_unused_parameters=True.')
            # Make model replica operate on the current device
            model = nn.parallel.DistributedDataParallel(
                module=model, device_ids=[cur_device], output_device=cur_device,
                find_unused_parameters=ddp_find_unused_parameters
            )

        check_random_seed_in_sync()

        misc.log_model_info(model)
        misc.check_pillow_performance()


        if args.load_epoch is not None:
            if load_exp.summary is None:
                load_exp.load_summary()
            if args.load_epoch == -1:
                load_epoch = int(load_exp.summary['epoch'].iloc[-1])
            elif args.load_epoch >= 0:
                load_epoch = args.load_epoch
            else:
                raise ValueError("Wrong args.load_epoch value: {:d}".format(args.load_epoch))

            weights_path = load_exp.get_checkpoint_path(load_epoch)
            checkpoint = loader.model_load_weights_GPU(model, weights_path, cur_device, world_size)
            start_epoch = load_epoch + 1

            # Crop summary. We don't need stats later than `start_epoch`
            exp.clip_summary_from_epoch(start_epoch)

        else:
            start_epoch = 0
            # Save training and validation stats

            if os.path.isfile(exp.summary_file):
                raise FileExistsError(f'Summary file "{exp.summary_file}" already exists and trying to initialise new experiment. Consider removing the experiment directory entirely to start new, or resume training by adding -l -1')

            # All processes have to check if exp.summary_file exists, before rank0 will create the file.
            # Otherwise, rank1 can see the empty file and throw an error after rank0 creates it.
            du.synchronize()  # before creating any experiment directory, we need to make sure that all versions for all processes is equal.

            if rank == 0:
                exp.init_summary()

            if hasattr(cfg, 'load_pretrained_postDDP'):
                cfg.load_pretrained_postDDP(model)


        criterions = cfg.dataset_cfg.task.get_criterions(cfg, splits)

        if rank == 0 and args.wandb_project is not None:
            with OutputLogger('wandb', input='stderr'):
                wandb.watch(model, criterion=criterions['train'], log='all')

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

        # Gradient Clipping
        if hasattr(cfg, 'clip_grad_max_norm'):
            clip_grad_max_norm = cfg.clip_grad_max_norm() if callable(cfg.clip_grad_max_norm) else cfg.clip_grad_max_norm
        else:
            clip_grad_max_norm = None

        # Scheduling
        scheduler = cfg.scheduler(optimiser, iters_per_epoch)
        scheduler_type = get_scheduler_type(scheduler)
        if scheduler_type == 'plateau':
            logger.info("ReduceLROnPlateau scheduler is selected. The `scheduler.step(val_loss)` function will be called at the end of epoch (after validation), but not every iteration.")
        elif scheduler_type == 'plateau_multiple':
            logger.info("ReduceLROnPlateauMultiple scheduler is selected. The `scheduler.step(val_loss, val_best_metric)` function will be called at the end of epoch (after validation), but not every iteration.")
        if scheduler_type.startswith('plateau'):
            if args.load_epoch is not None:
                if hasattr(cfg, 'load_scheduler_state'):
                    load_scheduler_state = cfg.load_scheduler_state
                else:
                    logger.info("cfg.load_scheduler_state not found. By default, it will try to load.")
                    load_scheduler_state = True
                if load_scheduler_state:
                    if checkpoint["scheduler_state"] is None:
                        logger.warning("Scheduler appears to have changed. Initialise using training stats (summary.csv) instead of the checkpoint.")
                        for i in range(start_epoch):
                            with OutputLogger(scheduler.__module__, "INFO"):   # redirect stdout print() to logging (for verbose=True)
                                scheduler.step(exp.summary['val_loss'][i])
                    else:
                        try:
                            logger.info("Loading scheduler state from the checkpoint.")
                            scheduler.load_state_dict(checkpoint["scheduler_state"])
                        except Exception:
                            logger.warning("Scheduler appears to have changed. Initialise using training stats (summary.csv) instead of the checkpoint.")
                            for i in range(start_epoch):
                                with OutputLogger(scheduler.__module__, "INFO"):   # redirect stdout print() to logging (for verbose=True)
                                    scheduler.step(exp.summary['val_loss'][i])
                else:
                    logger.info("NOT loading scheduler state from the checkpoint.")
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
                            logger.warning("Scheduler appears to have changed. Initialising using param last_epoch instead of using checkpoint scheduler state.")
                            scheduler = cfg.scheduler(optimiser, iters_per_epoch, last_epoch=last_iter_num-1)   # the last_epoch always has to be {the number of step() called} - 1
                        elif checkpoint["scheduler_state"]["last_epoch"] != last_iter_num:
                            logger.warning(f"Scheduler's last_epoch {checkpoint['scheduler_state']['last_epoch']} doesn't match with the epoch you're loading ({last_iter_num}). Possibly using different number of GPUs, batch size, or dataset size. Initialising using param last_epoch instead.")
                            scheduler = cfg.scheduler(optimiser, iters_per_epoch, last_epoch=last_iter_num-1)   # the last_epoch always has to be {the number of step() called} - 1
                        else:
                            try:
                                logger.info("Loading scheduler state from the checkpoint.")
                                scheduler.load_state_dict(checkpoint["scheduler_state"])
                            except Exception:
                                logger.warning("Scheduler appears to have changed. Initialising using param last_epoch instead of using checkpoint scheduler state.")
                                scheduler = cfg.scheduler(optimiser, iters_per_epoch, last_epoch=last_iter_num-1)   # the last_epoch always has to be {the number of step() called} - 1
                    else:
                        logger.info("Initialising scheduler using param last_epoch. NOT loading scheduler state from the checkpoint.")
                        scheduler = cfg.scheduler(optimiser, iters_per_epoch, last_epoch=last_iter_num-1)   # the last_epoch always has to be {the number of step() called} - 1



        #del checkpoint  # dereference seems crucial
        #torch.cuda.empty_cache()

        use_amp = search_expcfg_then_modelcfg('use_amp', False)
        if use_amp:
            logger.info('use_amp=True and this will speed up the training. If you encounter an error, consider updating the model to support AMP or setting this to False.')
        else:
            logger.info('use_amp=False. Consider setting it to True to speed up the training.')

        amp_scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        if use_amp:
            if args.load_epoch is not None:
                if "amp_scaler_state" in checkpoint.keys() and checkpoint["amp_scaler_state"] is not None:
                    logger.info("Loading Automatic Mixed Precision (AMP) state from the checkpoint.")
                    amp_scaler.load_state_dict(checkpoint["amp_scaler_state"])
                else:
                    logger.error("Could NOT load Automatic Mixed Precision (AMP) state from the checkpoint.")


        # to save maximum val_acc model, when option "args.save_mode" is "higher" or "last_and_peaks"
        # Also for ReduceLROnPlateauMultiple scheduling.
        best_metric, best_metric_fieldname = metrics.get_best_metric_and_fieldname()

        if rank == 0:
            tensorboard_writers = {}    # key: split
            def get_tensorboard_writer(split: str):
                if split not in tensorboard_writers.keys():
                    tensorboard_writers[split] = SummaryWriter(os.path.join(exp.tensorboard_runs_dir, split), comment=split)
                return tensorboard_writers[split]


            max_val_metric = None
            max_val_epoch = -1

            # Load the latest best metric until `start_epoch`
            if args.load_epoch is not None:
                all_stats_logical_idx = exp.summary['epoch'] < start_epoch
                all_stats = exp.summary[all_stats_logical_idx]
                for row in all_stats.itertuples():
                    val_metric = getattr(row, best_metric_fieldname)
                    is_better = True if max_val_metric is None else best_metric.is_better(val_metric, max_val_metric)
                    if is_better:
                        max_val_metric = val_metric
                        max_val_epoch = row.epoch

            # Plotting metrics
            metric_plotter = cfg.metric_plotter if hasattr(cfg, 'metric_plotter') else DefaultMetricPlotter()
            metric_plotter.add_metrics(metrics)
            telegram_reporter = cfg.telegram_reporter if hasattr(cfg, 'telegram_reporter') else DefaultTelegramReporter()

        # Because there are a lot of custom functions, we need to make sure that they're not doing anything wrong, by checking random values frequently!
        check_random_seed_in_sync()

        # Before training, check if it already meets the early stopping condition.
        early_stopping_flag = False
        if start_epoch > 0:
            if hasattr(cfg, 'early_stopping_condition'):
                if rank == 0:
                    metric_info = {'metrics': metrics,
                            'best_metric_fieldname': best_metric_fieldname, 'best_metric_is_better_func': best_metric.is_better}
                    early_stopping = cfg.early_stopping_condition(exp, metric_info)
                else:
                    early_stopping = -1   # dummy

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
            if rank == 0:
                exp.tg_send_text_with_expname('Early stopping triggered.')
        else:
            if rank == 0:
                if args.wandb_project is not None:
                    exp.tg_send_text_with_expname((f'Starting experiment..\n'
                        f'View W&B project at {wandb.run.get_project_url()}\n'
                        f'View W&B run at {wandb.run.get_url()}'))
                else:
                    exp.tg_send_text_with_expname('Starting experiment..')

            if hasattr(cfg, 'num_epochs'):
                num_epochs = cfg.num_epochs() if callable(cfg.num_epochs) else cfg.num_epochs
                logger.info(f'Training for maximum of {num_epochs} epochs')
            else:
                logger.warning('cfg.num_epochs not specified. By default it will train for 1000 epochs or until early stopping is triggered. args.num_epochs is deprecated and must not be used.')
                num_epochs = 1000

            for epoch in range(start_epoch, num_epochs):
                if hasattr(cfg, "epoch_start_script"):
                    epoch_start_script_start_time = time.time()
                    # structurise
                    train_kit = {}
                    train_kit["model"] = model
                    train_kit["optimiser"] = optimiser
                    train_kit["scheduler"] = scheduler
                    train_kit["criterions"] = criterions
                    train_kit["train_dataloader"] = train_dataloader
                    train_kit["data_unpack_funcs"] = data_unpack_funcs
                    train_kit["input_reshape_funcs"] = input_reshape_funcs
                    # train_kit can be modified in the function
                    cfg.epoch_start_script(epoch, copy.deepcopy(exp), copy.deepcopy(args), rank, world_size, train_kit)
                    # unpack and apply the modifications
                    model = train_kit["model"]
                    optimiser = train_kit["optimiser"]
                    scheduler = train_kit["scheduler"]
                    criterions = train_kit["criterions"]
                    train_dataloader = train_kit["train_dataloader"]
                    data_unpack_funcs = train_kit["data_unpack_funcs"]
                    input_reshape_funcs = train_kit["input_reshape_funcs"]

                    epoch_start_script_elapsed_time = time.time() - epoch_start_script_start_time
                else:
                    epoch_start_script_elapsed_time = 0.

                # Shuffle the dataset.
                loader.shuffle_dataset(train_dataloader, epoch, args.seed)

                if rank == 0:
                    print()
                    logger.info("Epoch %d/%d" % (epoch, num_epochs-1))

                sample_seen, total_samples, loss, elapsed_time, lr_epoch_end = train_epoch(model, optimiser, scheduler, criterions['train'], clip_grad_max_norm, use_amp, amp_scaler, train_dataloader, data_unpack_funcs['train'], metrics['train'], args.training_speed, rank, world_size, input_reshape_func=input_reshape_funcs['train'], refresh_period=args.refresh_period)
                if rank == 0:#{{{
                    elapsed_time += epoch_start_script_elapsed_time     # Include epoch start script as training time
                    curr_stat = {'epoch': epoch, 'train_runtime_sec': elapsed_time, 'train_loss': loss}
                    wandb_stat = {'Loss/train': loss, 'Runtime_sec/train': elapsed_time, 'Sample_seen/train': sample_seen, 'Total_samples/train': total_samples, 'LR/epoch_end': lr_epoch_end}
                    get_tensorboard_writer('train').add_scalar('Loss', loss, epoch)
                    get_tensorboard_writer('train').add_scalar('Runtime_sec', elapsed_time, epoch)
                    get_tensorboard_writer('train').add_scalar('Sample_seen', sample_seen, epoch)
                    get_tensorboard_writer('train').add_scalar('Total_samples', total_samples, epoch)
                    get_tensorboard_writer('train').add_scalar('LR_epoch_end', lr_epoch_end, epoch)

                    for metric in metrics['train']:
                        tensorboard_tags = metric.tensorboard_tags()
                        if not isinstance(tensorboard_tags, (list, tuple)):
                            tensorboard_tags = (tensorboard_tags,)
                        csv_fieldnames = metric.get_csv_fieldnames()
                        if not isinstance(csv_fieldnames, (list, tuple)):
                            csv_fieldnames = (csv_fieldnames,)
                        last_calculated_metrics = metric.last_calculated_metrics
                        if not isinstance(last_calculated_metrics, (list, tuple)):
                            last_calculated_metrics = (last_calculated_metrics,)

                        for tensorboard_tag, csv_fieldname, last_calculated_metric in zip(tensorboard_tags, csv_fieldnames, last_calculated_metrics):
                            if tensorboard_tag is not None:
                                # You can't assume that metrics in 'train' split are always called 'train'.
                                # Users can set custom name if they want to.
                                get_tensorboard_writer(metric.split).add_scalar(tensorboard_tag, last_calculated_metric, epoch)
                                wandb_stat[f'{tensorboard_tag}/{metric.split}'] = last_calculated_metric
                            if csv_fieldname is not None:
                                curr_stat[csv_fieldname] = last_calculated_metric


                    #}}}

                val_sample_seen, val_total_samples, val_loss, val_elapsed_time, _ = eval_epoch(model, criterions['val'], val_dataloader, data_unpack_funcs['val'], metrics['val'], best_metric, cfg.dataset_cfg.num_classes, 'val', rank, world_size, input_reshape_func=input_reshape_funcs['val'], scheduler=scheduler, refresh_period=args.refresh_period)
                if rank == 0:#{{{
                    curr_stat.update({'val_runtime_sec': val_elapsed_time, 'val_loss': val_loss})
                    wandb_stat.update({'Loss/val': val_loss, 'Runtime_sec/val': val_elapsed_time, 'Sample_seen/val': val_sample_seen, 'Total_samples/val': val_total_samples})
                    get_tensorboard_writer('val').add_scalar('Loss', val_loss, epoch)
                    get_tensorboard_writer('val').add_scalar('Runtime_sec', val_elapsed_time, epoch)
                    get_tensorboard_writer('val').add_scalar('Sample_seen', val_sample_seen, epoch)
                    get_tensorboard_writer('val').add_scalar('Total_samples', val_total_samples, epoch)


                    for metric in metrics['val']:
                        tensorboard_tags = metric.tensorboard_tags()
                        if not isinstance(tensorboard_tags, (list, tuple)):
                            tensorboard_tags = (tensorboard_tags,)
                        csv_fieldnames = metric.get_csv_fieldnames()
                        if not isinstance(csv_fieldnames, (list, tuple)):
                            csv_fieldnames = (csv_fieldnames,)
                        last_calculated_metrics = metric.last_calculated_metrics
                        if not isinstance(last_calculated_metrics, (list, tuple)):
                            last_calculated_metrics = (last_calculated_metrics,)

                        for tensorboard_tag, csv_fieldname, last_calculated_metric in zip(tensorboard_tags, csv_fieldnames, last_calculated_metrics):
                            if tensorboard_tag is not None:
                                get_tensorboard_writer(metric.split).add_scalar(tensorboard_tag, last_calculated_metric, epoch)
                                wandb_stat[f'{tensorboard_tag}/{metric.split}'] = last_calculated_metric
                            if csv_fieldname is not None:
                                curr_stat[csv_fieldname] = last_calculated_metric

                    #}}}

                if perform_multicropval and epoch % args.multi_crop_val_period == args.multi_crop_val_period -1:
                    multi_crop_val_sample_seen, multi_crop_val_total_samples, multi_crop_val_loss, multi_crop_val_elapsed_time, _ = eval_epoch(model, criterions['multicropval'], multi_crop_val_dataloader, data_unpack_funcs['multicropval'], metrics['multicropval'], None, cfg.dataset_cfg.num_classes, 'multicropval', rank, world_size, input_reshape_func=input_reshape_funcs['multicropval'], scheduler=None, refresh_period=args.refresh_period)  # No scheduler needed for multicropval
                    if rank == 0:#{{{
                        curr_stat.update({'multicropval_runtime_sec': multi_crop_val_elapsed_time, 'multicropval_loss': multi_crop_val_loss})
                        wandb_stat.update({'Loss/multicropval': multi_crop_val_loss, 'Runtime_sec/multicropval': multi_crop_val_elapsed_time,
                            'Sample_seen/multicropval': multi_crop_val_sample_seen, 'Total_samples/multicropval': multi_crop_val_total_samples})
                        get_tensorboard_writer('multicropval').add_scalar('Loss', multi_crop_val_loss, epoch)
                        get_tensorboard_writer('multicropval').add_scalar('Runtime_sec', multi_crop_val_elapsed_time, epoch)
                        get_tensorboard_writer('multicropval').add_scalar('Sample_seen', multi_crop_val_sample_seen, epoch)
                        get_tensorboard_writer('multicropval').add_scalar('Total_samples', multi_crop_val_total_samples, epoch)

                        for metric in metrics['multicropval']:
                            tensorboard_tags = metric.tensorboard_tags()
                            if not isinstance(tensorboard_tags, (list, tuple)):
                                tensorboard_tags = (tensorboard_tags,)
                            csv_fieldnames = metric.get_csv_fieldnames()
                            if not isinstance(csv_fieldnames, (list, tuple)):
                                csv_fieldnames = (csv_fieldnames,)
                            last_calculated_metrics = metric.last_calculated_metrics
                            if not isinstance(last_calculated_metrics, (list, tuple)):
                                last_calculated_metrics = (last_calculated_metrics,)

                            for tensorboard_tag, csv_fieldname, last_calculated_metric in zip(tensorboard_tags, csv_fieldnames, last_calculated_metrics):
                                if tensorboard_tag is not None:
                                    get_tensorboard_writer(metric.split).add_scalar(tensorboard_tag, last_calculated_metric, epoch)
                                    wandb_stat[f'{tensorboard_tag}/{metric.split}'] = last_calculated_metric
                                if csv_fieldname is not None:
                                    curr_stat[csv_fieldname] = last_calculated_metric


                        #}}}


                early_stopping = False      # need it for the entire process, because later we'll broadcast

                if rank == 0:
                    exp.add_summary_line(curr_stat)
                    if args.wandb_project is not None:
                        wandb.log(wandb_stat, step=epoch)

                    figs = metric_plotter.plot(exp)

                    if hasattr(cfg, 'early_stopping_condition'):
                        metric_info = {'metrics': metrics,
                                'best_metric_fieldname': best_metric_fieldname, 'best_metric_is_better_func': best_metric.is_better}
                        early_stopping = cfg.early_stopping_condition(exp, metric_info)

                    send_telegram = early_stopping or (epoch % args.telegram_post_period == args.telegram_post_period -1)
                    if send_telegram:
                        try:
                            logger.info('Sending plots to Telegram.')
                            start_time_plot = time.time()
                            telegram_reporter.report(metrics, exp, figs)
                        except (SSLCertVerificationError, OSError, NewConnectionError, MaxRetryError, ConnectionError):
                            """Usually, max-retries exceeds when you run it for a while.
                            Therefore, even if it fails to send the plots, don't crash the programme and keep it running.
                            """
                            logger.exception('Failed to send plots to Telegram.')

                        elapsed_time_plot = time.time() - start_time_plot

                    for plot_basename, fig in figs:
                        plt.close(fig)

                    if args.telegram_post_period < 100 and send_telegram and elapsed_time_plot > 5 * 60:
                        # Sending plots to telegram took longer than 5 mins.
                        # Telegram is limiting the traffic due to heavy usage.
                        logger.warning('Sending plots to Telegram took over 5 mins. Setting the plotting period to 100 epochs.')
                        args.telegram_post_period = 100


                    val_metric = curr_stat[best_metric_fieldname]    # which metric to use for deciding the best model

                    is_better = True if max_val_metric is None else best_metric.is_better(val_metric, max_val_metric)
                    if is_better:
                        max_val_metric = val_metric
                        max_val_epoch = epoch
                        if args.wandb_project is not None:
                            with OutputLogger('wandb', input='stderr'):
                                wandb.run.summary['best_val_metric_fieldname'] = best_metric_fieldname
                                wandb.run.summary['best_val_metric_value'] = val_metric
                                wandb.run.summary['best_val_epoch'] = max_val_epoch

                    if is_better or args.save_mode in ["all", "last_and_peaks"]:
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
                                        "scheduler_state": None if scheduler is None else scheduler.state_dict(),
                                        "amp_scaler_state": None if not use_amp else amp_scaler.state_dict(),
                                    }
                                torch.save(checkpoint, model_path)
                                io_error = False
                            except IOError:
                                logger.exception("IOError whilst saving the model.")
                                input("Press Enter to retry: ")

                    if args.save_mode == "last_and_peaks":
                        if epoch > 0 and epoch - 1 != max_val_epoch:
                            # delete previous checkpoint if not best
                            model_path = exp.get_checkpoint_path(epoch-1)
                            try:
                                logger.info(f"Removing previous model: {model_path}")
                                os.remove(model_path)
                            except IOError:
                                logger.exception("IOError whilst removing the model. Skipping..")

                    # Whatever the save mode is, make the best model symlink (keep it up-to-date)
                    best_model_name = exp.checkpoints_format.format(max_val_epoch)
                    best_symlink_path = os.path.join(exp.weights_dir, 'best.pth')
                    if os.path.islink(best_symlink_path):
                        os.remove(best_symlink_path)
                    os.symlink(best_model_name, best_symlink_path)  # first argument not full path -> make relative symlink

                    # Also make the last model symlink
                    last_model_name = exp.checkpoints_format.format(epoch)
                    last_symlink_path = os.path.join(exp.weights_dir, 'last.pth')
                    if os.path.islink(last_symlink_path):
                        os.remove(last_symlink_path)
                    os.symlink(last_model_name, last_symlink_path)  # first argument not full path -> make relative symlink


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

                check_random_seed_in_sync()


            logger.success('Finished training')
            if rank == 0:
                if args.wandb_project is not None:
                    exp.tg_send_text_with_expname((f'Finished training.\n'
                        f'View W&B project at {wandb.run.get_project_url()}\n'
                        f'View W&B run at {wandb.run.get_url()}'))

                    logs_artifact = wandb.Artifact('logs', type='logs')
                    logs_artifact.add_dir(exp.logs_dir)
                    wandb.log_artifact(logs_artifact)

                    if args.wandb_upload_models == 'best_and_last':
                        best_model_artifact = wandb.Artifact('best', type='weights')
                        best_model_artifact.add_file(os.path.join(exp.weights_dir, best_model_name))
                        wandb.log_artifact(best_model_artifact)
                        last_model_artifact = wandb.Artifact('last', type='weights')
                        last_model_artifact.add_file(os.path.join(exp.weights_dir, last_model_name))
                        wandb.log_artifact(last_model_artifact)
                else:
                    exp.tg_send_text_with_expname('Finished training.')

    except Exception:
        logger.exception("Exception occurred whilst training")
        # Every process is going to send exception.
        # This can make your Telegram report filled with many duplicates,
        # but at the same time it ensures that you receive a message when anything wrong happens.
#        if rank == 0:
        exp.tg_send_text_with_expname('Exception occurred whilst training\n\n' + traceback.format_exc())
