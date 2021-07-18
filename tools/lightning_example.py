import os
import time
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from torchmetrics import Accuracy
from pytorch_lightning.callbacks import Callback


from torch.optim.lr_scheduler import ReduceLROnPlateau

import coloredlogs, logging, verboselogs
logger = verboselogs.VerboseLogger(__name__)    # add logger.success

from experiment_utils.argparse_utils import add_exp_arguments
import argparse
def add_distributed_args(parser):
    """
    Parse the following arguments for the video training and testing pipeline.
    Args:
        shard_id (int): shard id for the current machine. Starts from 0 to
            num_shards - 1. If single machine is used, then set shard id to 0.
        num_shards (int): number of shards using by the job.
        init_method (str): initialization method to launch the job with multiple
            devices. Options includes TCP or shared file-system for
            initialization. details can be find in
            https://pytorch.org/docs/stable/distributed.html#tcp-initialization
        """
    parser.add_argument(
        "--local_world_size",
        help="Number of processes per machine. (i.e. number of GPUs to use per machine)",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--num_shards",
        help="Number of shards using by the job",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--init_method",
        help="Initialization method, includes TCP or shared file-system",
        default="tcp://localhost:19999",
        type=str,
    )

import dataset_configs
import model_configs
import exp_configs
from pyvideoai import config
def add_train_args(parser):
    parser.add_argument("-e", "--num_epochs", type=int, default=100, help="Number of epochs for training.")
    add_exp_arguments(parser, 
            root_default=config.DEFAULT_EXPERIMENT_ROOT, dataset_default='hmdb', model_default='i3d_resnet50', name_default='crop224_8x8_largejit_plateau_1scrop5tcrop_split1',
            dataset_channel_choices=dataset_configs.available_channels, model_channel_choices=model_configs.available_channels, exp_channel_choices=exp_configs.available_channels)
    parser.add_argument("-s", "--save_mode", type=str, default="last_and_peaks", choices=["all", "higher", "last_and_peaks"],  help="Checkpoint saving condition. all: save all epochs, higher: save whenever the highest validation performance model is found, last_and_peaks: save all epochs, but remove previous epoch if that wasn't the best.")
    parser.add_argument("-S", "--training_speed", type=str, default="standard", choices=["standard", "faster"],  help="Only applicable when using distributed multi-GPU training. 'faster' skips multiprocess commuication and CPU-GPU synchronisation for calculating training metrics (loss, accuracy) so they will be reported as 0. This probably won't give you any benefit on a single node, but for multi-node, it depends on the internet connection.")
    parser.add_argument("-l", "--load_epoch", type=int, default=None, help="Load from checkpoint. Set to -1 to load from the last checkpoint.")
    parser.add_argument("--seed", type=int, default=12, help="Random seed for np, torch, torch.cuda, DALI.")
    parser.add_argument("-t", "--multi_crop_val_period", type=int, default=-1, help="Number of epochs after full multi-crop validation is performed.")
    parser.add_argument("-T", "--telegram_post_period", type=int, default=10, help="Period (in epochs) to send the training stats on Telegram.")
    parser.add_argument("-B", "--telegram_bot_idx", type=int, default=0, help="Which Telegram bot to use defined in key.ini?")
    parser.add_argument("-w", "--dataloader_num_workers", type=int, default=4, help="num_workers for PyTorch Dataset loader.")



def get_parser():
    parser = argparse.ArgumentParser(description="Train and validate an action model",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_distributed_args(parser)
    add_train_args(parser)
    return parser

# https://github.com/PyTorchLightning/pytorch-lightning/discussions/6454
from pytorch_lightning.plugins.environments.cluster_environment import ClusterEnvironment
from pytorch_lightning.plugins.training_type import DDPPlugin
from typing import Any, Dict, List, Optional, Union

MY_ADDR = '127.0.0.1'
class CustomEnvironment(ClusterEnvironment):
    def __init__(self, num_nodes=1):
        super().__init__()
        self._num_nodes = num_nodes
        self._master_port = None

    def master_address(self):
        if self._num_nodes > 1:
            MASTER_ADDR = MY_ADDR
        else:
            MASTER_ADDR = os.environ.get("MASTER_ADDR", "127.0.0.1")
        #log.debug(f"MASTER_ADDR: {MASTER_ADDR}")
        return MASTER_ADDR

    def master_port(self):
        if self._master_port is None:
            if self._num_nodes > 1:
                self._master_port = MY_PORT
            else:
                #self._master_port = os.environ.get("MASTER_PORT", find_free_network_port())
                self._master_port = os.environ.get("MASTER_PORT", '5910')
        #log.debug(f"MASTER_PORT: {self._master_port}")
        return int(self._master_port)

    def world_size(self):
        #return None
        return os.environ.get("WORLD_SIZE", '1')

    def node_rank(self):
        #log.debug(f"NODE_RANK: {MY_RANK}")
        #return int(MY_RANK)
        return os.environ.get("NODE_RANK", '0')

    def local_rank(self) -> int:
        LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
        #log.debug(f"local_rank: {LOCAL_RANK}")
        return LOCAL_RANK


class ClusterDDPPlugin(DDPPlugin):
    def __init__(self, num_nodes=1, **kwargs: Union[Any, Dict[str, Any]]) -> None:
        super().__init__(
            num_nodes=num_nodes,
            cluster_environment=CustomEnvironment(),
            **kwargs,
        )

def get_lr(optimiser):
    ''' Works ONLY IF there's one parameter group only.
    Usually there's multiple groups with different learning rate.
    '''
    for param_group in optimiser.param_groups:
        return param_group['lr']

class TimeCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        pl_module.train_start_time = time.time()
    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.train_elapsed_time = time.time() - pl_module.train_start_time

class LightningModel(pl.LightningModule):

    def __init__(self, model, optimiser, scheduler, criterion, data_unpack_funcs, input_reshape_funcs, train_total_samples, train_total_iters):
        super().__init__()
        self.model = model
        self.optimiser = optimiser
        self.scheduler = scheduler 
        self.criterion = criterion
        self.data_unpack_funcs = data_unpack_funcs
        self.input_reshape_funcs = input_reshape_funcs

        self.train_total_samples = train_total_samples
        self.train_total_iters = train_total_iters

        self.train_acc = Accuracy()     # resets automatically every epoch
        self.val_acc = Accuracy()       # resets automatically every epoch

        self._init_trainval_variables()

    def _init_trainval_variables(self):
        self.train_sample_seen = torch.LongTensor([0])

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        outputs = self.model(x)
        return outputs 

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        inputs, uids, labels, _ = self.data_unpack_funcs['train'](batch)
        curr_batch_size = torch.LongTensor([labels.shape[0]])
        self.train_sample_seen += curr_batch_size
        #inputs, uids, labels, curr_batch_size = misc.data_to_gpu(inputs, uids, labels)
        if self.input_reshape_funcs['train']:
            inputs = input_reshape_funcs['train'](inputs)
        outputs = self.model(inputs)
        loss = criterion(outputs, labels)
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log('sample_seen', self.train_sample_seen, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.train_acc(outputs.softmax(dim=-1), labels)        
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'curr_batch_size': curr_batch_size}
    
    def training_epoch_end(self, training_step_outputs):
        num_iters = len(training_step_outputs)
#        for out in training_step_outputs:
#            print(out)

        if self.scheduler is not None and not isinstance(self.scheduler, ReduceLROnPlateau):
            lr = self.scheduler.get_last_lr()[0]
        else:
            lr = get_lr(self.optimiser)

#        write_str = " Train Iter: {:4d}/{:4d} - Sample: {:6d}/{:6d} - {:d}s - lr: {:.8f} - loss: {:.4f} - {:s}".format(num_iters, self.train_total_iters, self.train_sample_seen, self.train_total_samples, round(self.train_elapsed_time), lr, loss, final_logging_msg)
#        logger.info(write_str)

        self._init_trainval_variables()

    def validation_step(self, batch, batch_idx):
        inputs, uids, labels, _ = self.data_unpack_funcs['val'](batch)
        #inputs, uids, labels, curr_batch_size = misc.data_to_gpu(inputs, uids, labels)
        if self.input_reshape_funcs['val']:
            inputs = input_reshape_funcs['val'](inputs)
        outputs = self.model(inputs)
        loss = criterion(outputs, labels)
        # logs metrics for each validation_step,
        # and the average across the epoch, to the progress bar and logger
        # Add sync_dist=True to sync logging across all GPU workers (Only for validation and test?)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist = True)
        self.val_acc(outputs.softmax(dim=-1), labels)        
        self.log('val_acc', self.val_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        if self.scheduler is None:
            return self.optimiser

        optimisers = [self.optimiser]
        if isinstance(self.scheduler, ReduceLROnPlateau):
            schedulers = [{'scheduler': self.scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1,
                'strict': False,    # if True, stop training if val_loss is not available
                }]
        else:
            schedulers = [{'scheduler': self.scheduler,
                'interval': 'step',
                'frequency': 1,
                }]
        return optimisers, schedulers 


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    world_size = args.local_world_size * args.num_shards
    #
    pl.seed_everything(args.seed, workers=True)

    coloredlogs.install(fmt='%(name)s: %(lineno)4d - %(levelname)s - %(message)s', level='INFO')

    cfg = exp_configs.load_cfg(args.dataset, args.model, args.experiment_name, args.dataset_channel, args.model_channel, args.experiment_channel)

    splits = ['train', 'val']
    torch_datasets = {split: cfg.get_torch_dataset(split) for split in splits}
    data_unpack_funcs = {split: cfg.get_data_unpack_func(split) for split in splits}
    input_reshape_funcs= {split: cfg.get_input_reshape_func(split) for split in splits}
    batch_size = cfg.batch_size() if callable(cfg.batch_size) else cfg.batch_size
    if hasattr(cfg, 'val_batch_size'):
        val_batch_size = cfg.val_batch_size() if callable(cfg.val_batch_size) else cfg.val_batch_size
    else:
        val_batch_size = batch_size

    logger.info(f'Using batch size of {batch_size} per process (per GPU), resulting in total size of {batch_size * world_size}.')
    logger.info(f'Using validation batch size of {val_batch_size} per process (per GPU), resulting in total size of {val_batch_size * world_size}.')

    train_dataloader = torch.utils.data.DataLoader(torch_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=args.dataloader_num_workers, pin_memory=True, drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(torch_datasets['val'], batch_size=val_batch_size, shuffle=False, num_workers=args.dataloader_num_workers, pin_memory=True, drop_last=False)

    # Network
    # Construct the model
    model = cfg.load_model()
    if hasattr(cfg, 'get_optim_policies'):
        policies = cfg.get_optim_policies(model)
    elif hasattr(cfg.model_cfg, 'get_optim_policies'):
        policies = cfg.model_cfg.get_optim_policies(model)
    else:
        policies = model.parameters()

    criterion = cfg.dataset_cfg.task.get_criterion(cfg)
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
            if hasattr(cfg, 'load_scheduler_state'):
                load_scheduler_state = cfg.load_scheduler_state
            else:
                logger.info("cfg.load_scheduler_state not found. By default, it will try to load.")
                load_scheduler_state = True
            if load_scheduler_state:
                if checkpoint["scheduler_state"] is None:
                    logger.warning(f"Scheduler appears to have changed. Initialise using training stats (summary.csv) instead of the checkpoint.")
                    for i in range(start_epoch):
                        with OutputLogger(scheduler.__module__, "INFO"):   # redirect stdout print() to logging (for verbose=True)
                            scheduler.step(exp.summary['val_loss'][i])
                else:
                    try:
                        logger.info("Loading scheduler state from the checkpoint.")
                        scheduler.load_state_dict(checkpoint["scheduler_state"])
                    except Exception:
                        logger.warning(f"Scheduler appears to have changed. Initialise using training stats (summary.csv) instead of the checkpoint.")
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

    def search_expcfg_then_modelcfg(name, default):
        if hasattr(cfg, name):
            return getattr(cfg, name)
        elif hasattr(cfg.model_cfg, name):
            return getattr(cfg.model_cfg, name)
        return default

    use_amp = search_expcfg_then_modelcfg('use_amp', False)
    if use_amp:
        logger.info('use_amp=True and this will speed up the training. If you encounter an error, consider updating the model to support AMP or setting this to False.')
    else:
        logger.info('use_amp=False. Consider setting it to True to speed up the training.')


    pl_model = LightningModel(model, optimiser, scheduler, criterion, data_unpack_funcs, input_reshape_funcs,
            train_total_samples = len(train_dataloader.dataset),
            train_total_iters = len(train_dataloader))
    if world_size > 1:
        ddp_find_unused_parameters = search_expcfg_then_modelcfg('ddp_find_unused_parameters', True)
        if ddp_find_unused_parameters:
            logger.info('Will find unused parameters for distributed training. This introduces extra overheads, so set ddp_find_unused_parameters to True only when necessary.')
        else:
            logger.info('Will NOT find unused parameters for distributed training. If you see an error, consider setting ddp_find_unused_parameters=True.')
        trainer = pl.Trainer(gpus = args.local_world_size, num_nodes= args.num_shards, accelerator='ddp', precision=16 if use_amp else 32, plugins=DDPPlugin(find_unused_parameters=ddp_find_unused_parameters, num_nodes=args.num_shards),
                callbacks = [TimeCallback()],
                max_epochs = args.num_epochs)
    else:
        trainer = pl.Trainer(gpus=1, precision=16 if use_amp else 32,
                callbacks = [TimeCallback()],
                max_epochs = args.num_epochs)
    trainer.fit(pl_model, train_dataloader, val_dataloader)


