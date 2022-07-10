import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from .utils import distributed as du
from .utils import misc
import time
import sys
from .utils.distributed_sampler_for_val import count_true_samples, get_last_batch_size_singleGPU

from .utils.stdout_logger import OutputLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .utils.lr_scheduling import ReduceLROnPlateauMultiple, GradualWarmupScheduler


from .utils.losses.proselflc import ProSelfLC, MaskedProSelfLC

import logging
logger = logging.getLogger(__name__)

def get_lr(optimiser):
    ''' Works ONLY IF there's one parameter group only.
    Usually there's multiple groups with different learning rate.
    '''
    for param_group in optimiser.param_groups:
        return param_group['lr']

def get_scheduler_type(scheduler):
    """
    See type of scheduler (if plateau or not).
    If warmup scheduler, see what after_scheduler is.

    Returns:
        'none', 'others', 'plateau', 'plateau_multiple'
    """
    if scheduler is None:
        return 'none'
    elif isinstance(scheduler, ReduceLROnPlateau):
        return 'plateau'
    elif isinstance(scheduler, ReduceLROnPlateauMultiple):
        return 'plateau_multiple'
    else:
        if isinstance(scheduler, GradualWarmupScheduler):
            if isinstance(scheduler.after_scheduler, ReduceLROnPlateau):
                return 'plateau'
            elif isinstance(scheduler.after_scheduler, ReduceLROnPlateauMultiple):
                return 'plateau_multiple'
        else:
            return 'others'

def train_iter(model, optimiser, scheduler, criterion, clip_grad_max_norm, use_amp, amp_scaler, data, data_unpack_func, train_metrics, batch_size, speed = 'standard', start_time=None, it=None, total_iters=None, sample_seen=None, total_samples=None, loss_accum=None, rank = 0, world_size = 1, input_reshape_func = None, max_log_length=0, refresh_period=1):
    inputs, uids, labels, _ = data_unpack_func(data)#{{{
    inputs, uids, labels, curr_batch_size = misc.data_to_gpu(inputs, uids, labels)

    scheduler_type = get_scheduler_type(scheduler)
    if scheduler_type not in ['none', 'plateau', 'plateau_multiple']:
        lr = scheduler.get_last_lr()[0]
    else:
        lr = get_lr(optimiser)

    # zero the parameter gradients
    optimiser.zero_grad()

    # forward + backward + optimise
    if input_reshape_func:
        inputs = input_reshape_func(inputs)
    with torch.cuda.amp.autocast(enabled=use_amp):
        outputs = model(inputs)
        batch_loss = criterion(outputs, labels)
        # some criterions require calling step() (ProSelfLC)
        if isinstance(criterion, (ProSelfLC, MaskedProSelfLC)):
            if criterion.counter == 'iteration':
                criterion.step()

    if clip_grad_max_norm is None:
        misc.check_nan_losses(batch_loss)

    amp_scaler.scale(batch_loss).backward()
    if clip_grad_max_norm is not None:
        # Unscales the gradients of optimiser's assigned params in-place
        amp_scaler.unscale_(optimiser)
        #torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_max_norm)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_max_norm, error_if_nonfinite=False)   # For PyTorch 1.9.0 and above

    # If gradient clipping happened, the optimiser's gradients are already unscaled, so amp_scaler.step does not unscale them
    amp_scaler.step(optimiser)
    amp_scaler.update()
    if scheduler_type not in ['none', 'plateau', 'plateau_multiple']:
        with OutputLogger(scheduler.__module__, "INFO"):   # redirect stdout print() to logging (for verbose=True)
            scheduler.step()

    # loss
    batch_loss_accum = batch_loss * curr_batch_size

    # sync
    if world_size > 1:
        if speed == 'standard':
            for tensor in [curr_batch_size, batch_loss_accum]:
                dist.reduce(tensor, dst=0)

            if train_metrics is not None and len(train_metrics) > 0:
                (uids,labels,outputs) = du.all_gather([uids, labels, outputs])
        elif speed == 'faster':
            curr_batch_size = batch_size * world_size   # just infer from what we know
            batch_loss_accum = 0
        else:
            raise NotImplementedError()

    if rank == 0:
        if speed == 'standard':
            # Copy the stats from GPU to CPU (sync point).
            curr_batch_size, batch_loss_accum = (
                    curr_batch_size.item(),
                    batch_loss_accum.item(),
                )

        # Recalculate batch loss using the gathered data
        batch_loss = batch_loss_accum / curr_batch_size

        sample_seen += curr_batch_size
        # loss
        loss_accum += batch_loss_accum
        loss = loss_accum / sample_seen

        if train_metrics is not None and len(train_metrics) > 0:
            logging_msgs = []
            for metric in train_metrics:
                if speed == 'standard':
                    metric.add_clip_predictions(uids, outputs, labels)
                if metric.logging_msg_iter() is not None:
                    metric.calculate_metrics()
                    logging_msg = metric.logging_msg_iter()
                    logging_msgs.append(logging_msg)

            final_logging_msg = ' - '.join(logging_msgs)
        else:
            final_logging_msg = ''


        elapsed_time = time.time() - start_time
#        eta = int((total_samples-sample_seen) * elapsed_time / sample_seen)
        eta = int((total_iters-(it+1)) * elapsed_time / (it+1))
        write_str = "\r Train Iter: {:4d}/{:4d} - Sample: {:6d}/{:6d} - ETA: {:4d}s - lr: {:.8f} - batch_loss: {:.4f} - loss: {:.4f} - {:s}".format(it+1, total_iters, sample_seen, total_samples, eta, lr, batch_loss, loss, final_logging_msg)

        # Make sure you overwrite the entire line. To do so, we pad empty space characters to the string.
        if max_log_length < len(write_str):
            max_log_length = len(write_str)
        else:
            # Pad empty spaces
            write_str += ' ' * (max_log_length - len(write_str))

        if it % refresh_period == 0:
            sys.stdout.write(write_str)
            sys.stdout.flush()
    else:
        loss = None
        elapsed_time = None
        max_log_length = None

    return sample_seen, loss_accum, loss, elapsed_time, lr, max_log_length#}}}

def train_epoch(model, optimiser, scheduler, criterion, clip_grad_max_norm, use_amp, amp_scaler, dataloader, data_unpack_func, train_metrics, speed = 'standard', rank = 0, world_size = 1, input_reshape_func = None, refresh_period=1):
    """Train for one epoch.

    Args:
        model: PyTorch model
        optimiser: PyTorch optimiser
        criterion: PyTorch loss criterion (e.g. nn.CrossEntropyLoss())
        dataloader (iterator): Mini-batch data iterator. Requires self.epoch_size variable.
        rank (int): Rank of the process in distributed training.
        world_size (int): Total number of processes in distributed training.

    Returns:
        tuple: sample_seen, total_samples, loss, elapsed_time
    """

    model.train()

    if rank == 0:#{{{
        # train
        sample_seen = 0
        total_samples = len(dataloader.dataset)
        loss_accum = 0.
        start_time = time.time()
        total_iters = len(dataloader)
        max_log_length = 0
        if train_metrics is not None and len(train_metrics) > 0:
            for metric in train_metrics:
                metric.clean_data()
    else:
        sample_seen = None
        total_samples = None
        loss_accum = None
        start_time = None
        total_iters = None
        max_log_length = None

    # In training, set pad_last_batch=True so that the shard size is always equivalent and it doesn't give you no batch for some processes at the last batch.
    for it, data in enumerate(dataloader):

        sample_seen, loss_accum, loss, elapsed_time, lr, max_log_length = train_iter(model, optimiser, scheduler,
                criterion, clip_grad_max_norm, use_amp, amp_scaler,
                data, data_unpack_func,
                train_metrics, dataloader.batch_size, speed, start_time,
                it, total_iters, sample_seen, total_samples, loss_accum,
                rank, world_size,
                input_reshape_func, max_log_length, refresh_period=refresh_period)

    # some criterions require calling step() (ProSelfLC)
    if isinstance(criterion, (ProSelfLC, MaskedProSelfLC)):
        if criterion.counter == 'epoch':
            criterion.step()

    if rank == 0:
        if train_metrics is not None and len(train_metrics) > 0:
            logging_msgs = []
            for metric in train_metrics:
                if metric.logging_msg_epoch() is not None:
                    metric.calculate_metrics()
                    logging_msg = metric.logging_msg_epoch()
                    logging_msgs.append(logging_msg)

            final_logging_msg = ' - '.join(logging_msgs)
        else:
            final_logging_msg = ''

        sys.stdout.write("\r")
        sys.stdout.flush()


        write_str = " Train Iter: {:4d}/{:4d} - Sample: {:6d}/{:6d} - {:d}s - lr: {:.8f} - loss: {:.4f} - {:s}".format(it+1, total_iters, sample_seen, total_samples, round(elapsed_time), lr, loss, final_logging_msg)

        # Make sure you overwrite the entire line. To do so, we pad empty space characters to the string.
        if max_log_length < len(write_str):
            max_log_length = len(write_str)
        else:
            # Pad empty spaces
            write_str += ' ' * (max_log_length - len(write_str))

        logger.info(write_str)

    lr_epoch_end = lr
    return sample_seen, total_samples, loss, elapsed_time, lr_epoch_end#}}}


def eval_epoch(model, criterion, dataloader, data_unpack_func, val_metrics, best_metric, num_classes, split = 'val', rank = 0, world_size = 1, input_reshape_func = None, scheduler=None, refresh_period = 1, PAD_VALUE = -1):
    """Test for one epoch.

    Args:
        model: PyTorch model
        criterion: PyTorch loss criterion (e.g. nn.CrossEntropyLoss())
        dataloader (iterator): Mini-batch data iterator. Requires self.epoch_size and self.num_iters variable.
        val_metrics (list of pyvideoai.metrics.Metric)
        best_metric (pyvideoai.metrics.Metric): Has to be a part of val_metrics. Only required when split='val' and scheduler is ReduceLROnPlateauMultiple. Used for the scheduling.
        rank (int): Rank of the process in distributed training.
        world_size (int): Total number of processes in distributed training.
        scheduler (torch scheduler): Only needed when the scheduler is ReduceLROnPlateau, and using one clip eval.
        PAD_VALUE (int): The value to be padded when each process has different number of batch size. These padded value will be removed anyway after all gathering.

    Returns:
        tuple: sample_seen, total_samples, loss, elapsed_time, eval_log_str
    """

    cur_device = torch.cuda.current_device()

    scheduler_type = get_scheduler_type(scheduler)
    is_scheduler_plateau = scheduler_type.startswith('plateau')

    with torch.no_grad():#{{{
        model.eval()

        # These two variables are needed for every process for ReduceLROnPlateau scheduling.
        sample_seen = 0
        loss_accum = 0.

        if rank == 0:
#            num_correct_preds = 0
            total_samples = len(dataloader.dataset)
            total_iters = len(dataloader)

            if val_metrics is not None and len(val_metrics) > 0:
                for metric in val_metrics:
                    metric.clean_data()
#            video_metrics = VideoMetrics()
#            #video_metrics.clean_data()     # new validation
            start_time = time.time()
            max_log_length = 0

            if split == 'val':
                eval_mode = "One-clip Eval"
            elif split == 'multicropval':
                eval_mode = "Multi-crop Eval"
            else:
                eval_mode = "Eval"
        """
        if world_size > 1:
            # Number of iterations can be different over processes. Some processes need to wait until others finish.
            # Gathering the number of iterations for training.
            num_iters = torch.LongTensor([dataloader.num_iters]).to(cur_device)
            (num_iters,) = du.all_gather([num_iters])
            max_num_iters = num_iters.max().item()
        else:
            max_num_iters = dataloader.num_iters
        """

        if world_size > 1:
            shard_size, num_iters, last_batch_size = count_true_samples(dataloader.sampler, dataloader.batch_size)
        else:
            shard_size, num_iters, last_batch_size = total_samples, total_iters, get_last_batch_size_singleGPU(total_samples, dataloader.batch_size)

        if rank == 0 :
            assert num_iters == total_iters, "Implementation error"

        #for it in range(max_num_iters):
        for it, data in enumerate(dataloader):
            inputs, uids, labels, _ = data_unpack_func(data)
            inputs, uids, labels, curr_batch_size = misc.data_to_gpu(inputs, uids, labels)

            perform_forward = it < num_iters - 1 or last_batch_size > 0     # not last batch or last batch size is at least 1

            if it == num_iters - 1:
                """last batch true data sampling"""

                #if last_batch_size > 0:
                inputs = inputs[:last_batch_size]
                labels = labels[:last_batch_size]
                uids = uids[:last_batch_size]

#                else:
#                    # If #GPUs > last total batch size, some GPUs will get 0 inputs.
#                    inputs = inputs[:1]
#                    labels = labels[:1]
#                    uids = uids[:1]
                curr_batch_size = torch.LongTensor([last_batch_size]).to(cur_device, non_blocking=True)

#                uids = torch.IntTensor([]).to(cur_device, non_blocking=True)
#                labels = torch.IntTensor([]).to(cur_device, non_blocking=True)

            # forward
            if perform_forward:
                if input_reshape_func:
                    inputs = input_reshape_func(inputs)
                outputs = model(inputs)
                batch_loss = criterion(outputs, labels)
                batch_loss_accum = batch_loss * curr_batch_size
            else:
                batch_loss_accum = torch.FloatTensor([0]).to(cur_device, non_blocking=True)

            # Gather data
            if world_size > 1:
                if val_metrics is not None and len(val_metrics) > 0:
                    (curr_batch_sizes,) = du.all_gather([curr_batch_size])
                    max_batch_size = curr_batch_sizes.max()

                    # Pad to make the data same size before all_gather
                    uids = nn.functional.pad(uids, (0,max_batch_size-curr_batch_size), value=PAD_VALUE)

                    if labels.dim() == 2:
                        # multilabel
                        labels = nn.functional.pad(labels, (0,0,0,max_batch_size-curr_batch_size), value=PAD_VALUE)
                    elif labels.dim() == 1:
                        # singlelabel
                        labels = nn.functional.pad(labels, (0,max_batch_size-curr_batch_size), value=PAD_VALUE)
                    else:
                        raise NotImplementedError('Label with dim not 1 or 2 not expected.')

                    if curr_batch_size == 0:
                        outputs = torch.ones((max_batch_size, num_classes), dtype=torch.float32, device=cur_device) * PAD_VALUE
                    else:
                        outputs = nn.functional.pad(outputs, (0,0,0,max_batch_size-curr_batch_size), value=PAD_VALUE)

                    # Communicate with the padded data
                    (uids_gathered,labels_gathered,outputs_gathered) = du.all_gather([uids, labels, outputs])

                    if rank == 0:
                        # Remove padding from the received data
                        no_pad_row_mask = []     # logical indices
                        for proc_batch_size in curr_batch_sizes:
                            no_pad_row_mask.extend([i < proc_batch_size.item() for i in range(max_batch_size)])

                        uids = uids_gathered[no_pad_row_mask]
                        labels = labels_gathered[no_pad_row_mask]
                        outputs = outputs_gathered[no_pad_row_mask]

                # Communicate other data
                for tensor in [curr_batch_size, batch_loss_accum]:
                    dist.reduce(tensor, dst=0)

            if rank == 0 or is_scheduler_plateau:
                # Copy the stats from GPU to CPU (sync point).
                curr_batch_size, batch_loss_accum = (
                        curr_batch_size.item(),
                        batch_loss_accum.item(),
                    )

                sample_seen += curr_batch_size
                # loss
                loss_accum += batch_loss_accum
                loss = loss_accum / sample_seen

            if rank == 0:
                if val_metrics is not None and len(val_metrics) > 0:
                    logging_msgs = []
                    for metric in val_metrics:
                        metric.add_clip_predictions(uids, outputs, labels)
                        logging_msg = metric.logging_msg_iter()
                        if logging_msg is not None:
                            metric.calculate_metrics()
                            logging_msg = metric.logging_msg_iter()
                            logging_msgs.append(logging_msg)

                    final_logging_msg = ' - '.join(logging_msgs)
                else:
                    final_logging_msg = ''

                elapsed_time = time.time() - start_time
                #eta = int((total_samples-sample_seen) * elapsed_time / sample_seen)
                eta = int((total_iters-(it+1)) * elapsed_time / (it+1))

                write_str = "\r {:s} Iter: {:4d}/{:4d} - Sample: {:6d}/{:6d} - ETA: {:4d}s - {:s}_loss: {:.4f} - {:s}".format(eval_mode, it+1, total_iters, sample_seen, total_samples, eta, split, loss, final_logging_msg)
                # Make sure you overwrite the entire line. To do so, we pad empty space characters to the string.
                if max_log_length < len(write_str):
                    max_log_length = len(write_str)
                else:
                    # Pad empty spaces
                    write_str += ' ' * (max_log_length - len(write_str))

                if it % refresh_period == 0:
                    sys.stdout.write(write_str)
                    sys.stdout.flush()


        # Reset the iterator. Needs to be done at the end of epoch when __next__ is directly called instead of doing iteration.
#        dataloader.reset()


    if rank == 0:
        sys.stdout.write("\r")
        sys.stdout.flush()

        if val_metrics is not None and len(val_metrics) > 0:
            logging_msgs = []
            for metric in val_metrics:
                logging_msg = metric.logging_msg_epoch()
                if logging_msg is not None:
                    metric.calculate_metrics()
                    logging_msg = metric.logging_msg_epoch()
                    logging_msgs.append(logging_msg)

            final_logging_msg = ' - '.join(logging_msgs)
        else:
            final_logging_msg = ''

        eval_log_str = " {:s} Iter: {:4d}/{:4d} - Sample: {:6d}/{:6d} - {:d}s - {:s}_loss: {:.4f} - {:s}".format(eval_mode, it+1, total_iters, sample_seen, total_samples, round(elapsed_time), split, loss, final_logging_msg)
        # Make sure you overwrite the entire line. To do so, we pad empty space characters to the string.
        if max_log_length < len(eval_log_str):
            max_log_length = len(eval_log_str)
        else:
            # Pad empty spaces
            eval_log_str += ' ' * (max_log_length - len(eval_log_str))

        logger.info(eval_log_str)



    # Update ReduceLROnPlateau scheduling
    if split == 'val':
        if rank == 0:
            if best_metric is not None:
                if best_metric.logging_msg_epoch() is None:
                    # Metric not calculated. Calculate now
                    best_metric.calculate_metrics()

        if scheduler_type == 'plateau':
            with OutputLogger(scheduler.__module__, "INFO"):   # redirect stdout print() to logging (for verbose=True)
                scheduler.step(metrics=loss)
        elif scheduler_type == 'plateau_multiple':
            """This gets complicated.
            Since all metrics are only calculated on rank 0, we have to broadcast the final calculated value across the world.
            """
            if rank == 0:
                last_best_metric = best_metric.last_calculated_metrics
                if isinstance(last_best_metric, (list,tuple)):
                    last_best_metric = last_best_metric[0]  # use the first one. Maybe it is top1 acc.

                last_best_metric = torch.FloatTensor([last_best_metric]).to(cur_device, non_blocking=True)
            else:
                last_best_metric = torch.FloatTensor([-100]).to(cur_device, non_blocking=True)

            if world_size > 1:
                dist.broadcast(last_best_metric, 0)
            last_best_metric = last_best_metric.item()

            with OutputLogger(scheduler.__module__, "INFO"):   # redirect stdout print() to logging (for verbose=True)
                scheduler.step(metrics=loss, metrics2=last_best_metric, metrics2_is_better=best_metric.is_better)


    if rank != 0:
        sample_seen = None
        total_samples = None
        loss = None
        elapsed_time = None
        eval_log_str = None

    return sample_seen, total_samples, loss, elapsed_time, eval_log_str#}}}

# NOTE: DEPRECATED
def test_epoch_DALI(model, criterion, dataloader, data_unpack_func, num_classes, rank = 0, world_size = 1, PAD_VALUE = -1):#{{{
    """Test for one epoch.

    Args:
        model: PyTorch model
        criterion: PyTorch loss criterion (e.g. nn.CrossEntropyLoss())
        dataloader (iterator): Mini-batch data iterator. Requires self.epoch_size and self.num_iters variable.
        rank (int): Rank of the process in distributed training.
        world_size (int): Total number of processes in distributed training.
        PAD_VALUE (int): The value to be padded when each process has different number of batch size. These padded value will be removed anyway after all gathering.

    Returns:
        tuple: sample_seen, total_samples, loss, acc, vid_acc_top1, vid_acc_top5, elapsed_time, video_metrics (VideoMetrics)
    """

    cur_device = torch.cuda.current_device()

    with torch.no_grad():
        model.eval()

        if rank == 0:
            sample_seen = 0
            num_correct_preds = 0
            loss_accum = 0.
            total_samples = len(dataloader.dataset)

            video_metrics = VideoMetrics()
            #video_metrics.clean_data()     # new validation
            start_time = time.time()

        """
        if world_size > 1:
            # Number of iterations can be different over processes. Some processes need to wait until others finish.
            # Gathering the number of iterations for training.
            num_iters = torch.LongTensor([dataloader.num_iters]).to(cur_device)
            (num_iters,) = du.all_gather([num_iters])
            max_num_iters = num_iters.max().item()
        else:
            max_num_iters = dataloader.num_iters
        """

        #for it in range(max_num_iters):
        for it, data in enumerate(dataloader):
            #if it < dataloader.num_iters:
            if True:
                """Still have data to load"""
                #data = next(dataloader)

                inputs, labels, uids, curr_batch_size, _ = data_unpack_func(data)
                if world_size > 1:
                    curr_batch_size = curr_batch_size.to(cur_device, non_blocking=True)

                # forward
                outputs = model(inputs)
                batch_loss = criterion(outputs, labels)
                batch_loss_accum = batch_loss * curr_batch_size

                _, predicted = torch.max(outputs.data, 1)
                batch_correct = (predicted == labels).sum()
            else:
                """No more data but waiting for other processes to finish"""
                uids = torch.IntTensor([]).to(cur_device, non_blocking=True)
                labels = torch.IntTensor([]).to(cur_device, non_blocking=True)
                #outputs
                curr_batch_size = torch.LongTensor([0]).to(cur_device, non_blocking=True)
                batch_loss_accum = torch.FloatTensor([0]).to(cur_device, non_blocking=True)
                batch_correct = torch.LongTensor([0]).to(cur_device, non_blocking=True)

            # Gather data
            if world_size > 1:
                (curr_batch_sizes,) = du.all_gather([curr_batch_size])
                max_batch_size = curr_batch_sizes.max()

                # Pad to make the data same size before all_gather
                uids = nn.functional.pad(uids, (0,max_batch_size-curr_batch_size), value=PAD_VALUE)
                labels = nn.functional.pad(labels, (0,max_batch_size-curr_batch_size), value=PAD_VALUE)
                if curr_batch_size == 0:
                    outputs = torch.ones((max_batch_size, num_classes), dtype=torch.float32, device=cur_device) * PAD_VALUE
                else:
                    outputs = nn.functional.pad(outputs, (0,0,0,max_batch_size-curr_batch_size), value=PAD_VALUE)

                # Communicate with the padded data
                (uids_gathered,labels_gathered,outputs_gathered) = du.all_gather([uids, labels, outputs])

                if rank == 0:
                    # Remove padding from the received data
                    no_pad_row_mask = []     # logical indices
                    for proc_batch_size in curr_batch_sizes:
                        no_pad_row_mask.extend([i < proc_batch_size.item() for i in range(max_batch_size)])

                    uids = uids_gathered[no_pad_row_mask]
                    labels = labels_gathered[no_pad_row_mask]
                    outputs = outputs_gathered[no_pad_row_mask]

                # Communicate other data
                for tensor in [curr_batch_size, batch_loss_accum, batch_correct]:
                    dist.reduce(tensor, dst=0)

            if rank == 0 or is_scheduler_plateau:
                # Copy the stats from GPU to CPU (sync point).
                curr_batch_size, batch_loss_accum, batch_correct = (
                        curr_batch_size.item(),
                        batch_loss_accum.item(),
                        batch_correct.item(),
                    )

                sample_seen += curr_batch_size
                # loss
                loss_accum += batch_loss_accum
                loss = loss_accum / sample_seen

            if rank == 0:
                # accuracy
                num_correct_preds += batch_correct

                # video accuracy top1, top5
                video_metrics.add_clip_predictions(uids, outputs, labels, apply_activation='softmax')

                acc = num_correct_preds / sample_seen
                elapsed_time = time.time() - start_time
                eta = int((total_samples-sample_seen) * elapsed_time / sample_seen)
                sys.stdout.write("\r {:6d}/{:6d} - ETA: {:4d}s - val_loss: {:.4f} - val_acc: {:.4f}        ".format(sample_seen, total_samples, eta, loss, acc))
                sys.stdout.flush()

        # Reset the iterator. Needs to be done at the end of epoch when __next__ is directly called instead of doing iteration.
#        dataloader.reset()

    if is_scheduler_plateau:
        with OutputLogger(scheduler.__module__, "INFO"):   # redirect stdout print() to logging (for verbose=True)
            scheduler.step(metrics=loss)

    if rank == 0:
        vid_acc_top1, vid_acc_top5 = video_metrics.accuracy(topk=(1,5))

        print("\r {:6d}/{:6d} - {:d}s - val_loss: {:.4f} - val_acc: {:.4f} - val_vid_acc_top1: {:.4f} - val_vid_acc_top5: {:.4f}      ".format(sample_seen, total_samples, round(elapsed_time), loss, acc, vid_acc_top1, vid_acc_top5))
    else:
        sample_seen = None
        total_samples = None
        loss = None
        acc = None
        vid_acc_top1 = None
        vid_acc_top5 = None
        elapsed_time = None
        video_metrics = None

    return sample_seen, total_samples, loss, acc, vid_acc_top1, vid_acc_top5, elapsed_time, video_metrics#}}}

def extract_features(model, dataloader, data_unpack_func, num_classes, split = 'val', rank = 0, world_size = 1, input_reshape_func = None, refresh_period = 1, PAD_VALUE = -1):
    """Extract features for all samples in the dataset.
    Note that model can output multiple features in tuple.

    Args:
        model: PyTorch model
        dataloader (iterator): Mini-batch data iterator. Requires self.epoch_size and self.num_iters variable.
        data_unpack_func: Should return (inputs, uids, labels, {'spatial_idx': spatial_idx, 'temporal_idx': temporal_idx, 'frame_indices': frame_indices})
        rank (int): Rank of the process in distributed training.
        world_size (int): Total number of processes in distributed training.
        PAD_VALUE (int): The value to be padded when each process has different number of batch size. These padded value will be removed anyway after all gathering.

    Returns:
        tuple: sample_seen, total_samples, elapsed_time, eval_log_str
    """

    cur_device = torch.cuda.current_device()

    with torch.no_grad():
        model.eval()

        sample_seen = 0

        if rank == 0:
            total_samples = len(dataloader.dataset)
            total_iters = len(dataloader)

            start_time = time.time()
            max_log_length = 0

            if split == 'val':
                extract_mode = "One-clip Feature Extraction"
            elif split == 'multicropval':
                extract_mode = "Multi-crop Feature Extraction"
            else:
                extract_mode = "Feature Extraction"

            feature_data = {'video_ids': [],
                    'labels': [],
                    'clip_features': None,  # model can output multiple features, hence should be list of lists
                    'spatial_indices': [],
                    'temporal_indices': [],
                    'frame_indices': [],
                    }

        if world_size > 1:
            shard_size, num_iters, last_batch_size = count_true_samples(dataloader.sampler, dataloader.batch_size)
        else:
            shard_size, num_iters, last_batch_size = total_samples, total_iters, get_last_batch_size_singleGPU(total_samples, dataloader.batch_size)

        if rank == 0 :
            assert num_iters == total_iters, "Implementation error"

        num_model_outputs: int = None       # To remember how many outputs model had.

        #for it in range(max_num_iters):
        for it, data in enumerate(dataloader):
            inputs, uids, labels, extra_info = data_unpack_func(data)
            spatial_idx = extra_info['spatial_idx']
            temporal_idx = extra_info['temporal_idx']
            frame_indices = extra_info['frame_indices']

            inputs, uids, labels, spatial_idx, temporal_idx, frame_indices, curr_batch_size = misc.data_to_gpu(inputs, uids, labels, spatial_idx, temporal_idx, frame_indices)

            perform_forward = it < num_iters - 1 or last_batch_size > 0     # not last batch or last batch size is at least 1

            if it == num_iters - 1:
                """last batch true data sampling"""

                inputs = inputs[:last_batch_size]
                labels = labels[:last_batch_size]
                uids = uids[:last_batch_size]
                spatial_idx = spatial_idx[:last_batch_size]
                temporal_idx = temporal_idx[:last_batch_size]
                frame_indices = frame_indices[:last_batch_size]

                curr_batch_size = torch.LongTensor([last_batch_size]).to(cur_device, non_blocking=True)

            # forward
            if perform_forward:
                if input_reshape_func:
                    inputs = input_reshape_func(inputs)
                outputs = model(inputs)

                # update num_model_outputs, only the first time of iteration.
                if num_model_outputs is None:
                    if rank == 0:
                        if isinstance(outputs, tuple):
                            if world_size > 1:
                                num_model_outputs_dist = torch.LongTensor([len(outputs)]).to(cur_device, non_blocking=True)
                            else:
                                num_model_outputs = len(outputs)
                        else:
                            if world_size > 1:
                                num_model_outputs_dist = torch.LongTensor([1]).to(cur_device, non_blocking=True)
                            else:
                                num_model_outputs = 1

                    else:
                        num_model_outputs_dist = torch.LongTensor([-100]).to(cur_device, non_blocking=True)

                    if world_size > 1:
                        dist.broadcast(num_model_outputs_dist, 0)
                        num_model_outputs = num_model_outputs_dist.item()

                    if rank == 0:
                        feature_data['clip_features'] = tuple([] for _ in range(num_model_outputs))

                if num_model_outputs == 1:
                    outputs = tuple([outputs])  # if you convert directly like tuple(outputs,), it will split the tensor batch dimension into tuple. This case, we want the tuple size to be 1.

            # Gather data
            if world_size > 1:
                (curr_batch_sizes,) = du.all_gather([curr_batch_size])
                max_batch_size = curr_batch_sizes.max()

                # Pad to make the data same size before all_gather
                uids = nn.functional.pad(uids, (0,max_batch_size-curr_batch_size), value=PAD_VALUE)
                spatial_idx = nn.functional.pad(spatial_idx, (0,max_batch_size-curr_batch_size), value=PAD_VALUE)
                temporal_idx = nn.functional.pad(temporal_idx, (0,max_batch_size-curr_batch_size), value=PAD_VALUE)
                frame_indices = nn.functional.pad(frame_indices, (0,0,0,max_batch_size-curr_batch_size), value=PAD_VALUE)

                if labels.dim() == 2:
                    # multilabel
                    labels = nn.functional.pad(labels, (0,0,0,max_batch_size-curr_batch_size), value=PAD_VALUE)
                elif labels.dim() == 1:
                    # singlelabel
                    labels = nn.functional.pad(labels, (0,max_batch_size-curr_batch_size), value=PAD_VALUE)
                else:
                    raise NotImplementedError('Label with dim not 1 or 2 not expected.')

                if curr_batch_size == 0:
                    outputs = tuple(torch.ones((max_batch_size, num_classes), dtype=torch.float32, device=cur_device) * PAD_VALUE for _ in range(num_model_outputs))
                else:
                    outputs = tuple(nn.functional.pad(output, (0,0,0,max_batch_size-curr_batch_size), value=PAD_VALUE) for output in outputs)

                # Communicate with the padded data
                (uids_gathered,
                        labels_gathered,
                        spatial_idx_gathered,
                        temporal_idx_gathered,
                        frame_indices_gathered,
                        ) = du.all_gather([uids, labels, spatial_idx, temporal_idx, frame_indices])
                # outputs is a tuple of tensors
                outputs_gathered = du.all_gather(outputs)

                if rank == 0:
                    # Remove padding from the received data
                    no_pad_row_mask = []     # logical indices
                    for proc_batch_size in curr_batch_sizes:
                        no_pad_row_mask.extend([i < proc_batch_size.item() for i in range(max_batch_size)])

#                    print(f'{uids_gathered.shape = }')
#                    print(f'{labels_gathered.shape = }')
#                    print(f'{outputs_gathered.shape = }')
#                    print(f'{spatial_idx_gathered.shape = }')
#                    print(f'{temporal_idx_gathered.shape = }')
                    uids = uids_gathered[no_pad_row_mask]
                    labels = labels_gathered[no_pad_row_mask]
                    outputs = tuple(output_gathered[no_pad_row_mask] for output_gathered in outputs_gathered)
                    spatial_idx = spatial_idx_gathered[no_pad_row_mask]
                    temporal_idx = temporal_idx_gathered[no_pad_row_mask]
                    frame_indices = frame_indices_gathered[no_pad_row_mask]

                # Communicate other data
                dist.reduce(curr_batch_size, dst=0)

            if rank == 0:
                # Copy the stats from GPU to CPU (sync point).
                curr_batch_size = curr_batch_size.item()
                sample_seen += curr_batch_size

                feature_data['video_ids'].append(np.array(uids.cpu()))
                feature_data['labels'].append(np.array(labels.cpu()))
                for output_idx, output in enumerate(outputs):
                    feature_data['clip_features'][output_idx].append(np.array(output.cpu()))
                feature_data['spatial_indices'].append(np.array(spatial_idx.cpu()))
                feature_data['temporal_indices'].append(np.array(temporal_idx.cpu()))
                feature_data['frame_indices'].append(np.array(frame_indices.cpu()))

                elapsed_time = time.time() - start_time
                #eta = int((total_samples-sample_seen) * elapsed_time / sample_seen)
                eta = int((total_iters-(it+1)) * elapsed_time / (it+1))

                write_str = "\r {:s} Iter: {:4d}/{:4d} - Sample: {:6d}/{:6d} - ETA: {:4d}s".format(extract_mode, it+1, total_iters, sample_seen, total_samples, eta)
                # Make sure you overwrite the entire line. To do so, we pad empty space characters to the string.
                if max_log_length < len(write_str):
                    max_log_length = len(write_str)
                else:
                    # Pad empty spaces
                    write_str += ' ' * (max_log_length - len(write_str))

                if it % refresh_period == 0:
                    sys.stdout.write(write_str)
                    sys.stdout.flush()


        # Reset the iterator. Needs to be done at the end of epoch when __next__ is directly called instead of doing iteration.
#        dataloader.reset()


    if rank == 0:
        sys.stdout.write("\r")
        sys.stdout.flush()

        eval_log_str = " {:s} Iter: {:4d}/{:4d} - Sample: {:6d}/{:6d} - {:d}s".format(extract_mode, it+1, total_iters, sample_seen, total_samples, round(elapsed_time))
        # Make sure you overwrite the entire line. To do so, we pad empty space characters to the string.
        if max_log_length < len(eval_log_str):
            max_log_length = len(eval_log_str)
        else:
            # Pad empty spaces
            eval_log_str += ' ' * (max_log_length - len(eval_log_str))

        logger.info(eval_log_str)

        # convert list[np.array] into huge numpy array
        for key in feature_data.keys():
            if key == 'clip_features':
                feature_data[key] = tuple(np.concatenate(feature_list, axis=0) for feature_list in feature_data[key])
            else:
                feature_data[key] = np.concatenate(feature_data[key], axis=0)

    if rank != 0:
        feature_data = None
        sample_seen = None
        total_samples = None
        elapsed_time = None
        eval_log_str = None

    return feature_data, sample_seen, total_samples, elapsed_time, eval_log_str
