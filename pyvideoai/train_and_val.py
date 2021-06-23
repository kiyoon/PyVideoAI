import torch
from torch import nn
import torch.distributed as dist
from .utils import distributed as du
from .utils import misc
from .utils.video_metrics import VideoMetrics
import time
import sys
from .utils.distributed_sampler_for_val import count_true_samples

from .utils.stdout_logger import OutputLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau


import logging
logger = logging.getLogger(__name__)

'''
def unpack_data(data):
    inputs = [data["data"]]     # make it as a list for slowonly (not slowfast)
    labels = data["label"]
    uids = data["uid"]
    curr_batch_size = torch.LongTensor([labels.shape[0]])

    return inputs, labels, uids, curr_batch_size
'''

def get_lr(optimiser):
    ''' Works ONLY IF there's one parameter group only.
    Usually there's multiple groups with different learning rate.
    '''
    for param_group in optimiser.param_groups:
        return param_group['lr']

def train_iter(model, optimiser, scheduler, criterion, data, data_unpack_func, start_time=None, it=None, total_iters=None, sample_seen=None, num_correct_preds=None, total_samples=None, loss_accum=None, rank = 0, world_size = 1, input_reshape_func = None):
    inputs, uids, labels, _ = data_unpack_func(data)#{{{
    inputs, uids, labels, curr_batch_size = misc.data_to_gpu(inputs, uids, labels)

    if scheduler is not None and not isinstance(scheduler, ReduceLROnPlateau):
        lr = scheduler.get_last_lr()[0]
    else:
        lr = get_lr(optimiser)

    # zero the parameter gradients
    optimiser.zero_grad()

    # forward + backward + optimise
    if input_reshape_func:
        inputs = input_reshape_func(inputs)
    outputs = model(inputs)
    batch_loss = criterion(outputs, labels)
    misc.check_nan_losses(batch_loss)
    batch_loss.backward()
    optimiser.step()
    if scheduler is not None and not isinstance(scheduler, ReduceLROnPlateau):      # Plateau scheduler will update at the end of epoch after validation.
        with OutputLogger(scheduler.__module__, "INFO"):   # redirect stdout print() to logging (for verbose=True)
            scheduler.step()

    # loss
    batch_loss_accum = batch_loss * curr_batch_size

    # accuracy
    _, predicted = torch.max(outputs.data, 1)
    batch_correct = (predicted == labels).sum()

    # sync
    if world_size > 1:
        for tensor in [curr_batch_size, batch_loss_accum, batch_correct]:
            dist.reduce(tensor, dst=0)

    if rank == 0:
        # Copy the stats from GPU to CPU (sync point).
        curr_batch_size, batch_loss_accum, batch_correct = (
                curr_batch_size.item(),
                batch_loss_accum.item(),
                batch_correct.item(),
            )

        # Recalculate batch loss using the gathered data
        batch_loss = batch_loss_accum / curr_batch_size

        sample_seen += curr_batch_size
        # loss
        loss_accum += batch_loss_accum
        loss = loss_accum / sample_seen

        # accuracy
        num_correct_preds += batch_correct 
        batch_acc = batch_correct / curr_batch_size 
        acc = num_correct_preds / sample_seen

        elapsed_time = time.time() - start_time
#        eta = int((total_samples-sample_seen) * elapsed_time / sample_seen)
        eta = int((total_iters-(it+1)) * elapsed_time / (it+1))
        sys.stdout.write("\r Train Iter: {:4d}/{:4d} - Sample: {:6d}/{:6d} - ETA: {:4d}s - lr: {:.8f} - batch_loss: {:.4f} - loss: {:.4f} - batch_acc: {:.4f} - acc: {:.4f}        ".format(it+1, total_iters, sample_seen, total_samples, eta, lr, batch_loss, loss, batch_acc, acc))
        sys.stdout.flush()
    else:
        loss = None
        acc = None
        elapsed_time = None

    return sample_seen, num_correct_preds, loss_accum, loss, acc, elapsed_time, lr#}}}

def train_epoch(model, optimiser, scheduler, criterion, dataloader, data_unpack_func, rank = 0, world_size = 1, input_reshape_func = None):
    """Train for one epoch.

    Args:
        model: PyTorch model
        optimiser: PyTorch optimiser
        criterion: PyTorch loss criterion (e.g. nn.CrossEntropyLoss())
        dataloader (iterator): Mini-batch data iterator. Requires self.epoch_size variable.
        rank (int): Rank of the process in distributed training.
        world_size (int): Total number of processes in distributed training.
    
    Returns:
        tuple: sample_seen, total_samples, loss, acc, elapsed_time
    """

    model.train()

    if rank == 0:#{{{
        # train
        sample_seen = 0
        num_correct_preds = 0
        total_samples = len(dataloader.dataset)
        loss_accum = 0.
        start_time = time.time()
        total_iters = len(dataloader)
    else:
        sample_seen = None
        num_correct_preds = None
        total_samples = None
        loss_accum = None
        start_time = None
        total_iters = None

    # In training, set pad_last_batch=True so that the shard size is always equivalent and it doesn't give you no batch for some processes at the last batch.
    for it, data in enumerate(dataloader):

        sample_seen, num_correct_preds, loss_accum, loss, acc, elapsed_time, lr = train_iter(model, optimiser, scheduler, criterion, data, data_unpack_func, start_time, it, total_iters, sample_seen, num_correct_preds, total_samples, loss_accum, rank, world_size, input_reshape_func)

    if rank == 0:
        sys.stdout.write("\r")
        sys.stdout.flush()
        logger.info(" Train Iter: {:4d}/{:4d} - Sample: {:6d}/{:6d} - {:d}s - lr: {:.8f} - loss: {:.4f} - acc: {:.4f}                                                            ".format(it+1, total_iters, sample_seen, total_samples, round(elapsed_time), lr, loss, acc))

    return sample_seen, total_samples, loss, acc, elapsed_time#}}}


def eval_epoch(model, criterion, dataloader, data_unpack_func, num_classes, batch_size, one_clip = False, rank = 0, world_size = 1, input_reshape_func = None, scheduler=None, PAD_VALUE = -1):
    """Test for one epoch.

    Args:
        model: PyTorch model
        criterion: PyTorch loss criterion (e.g. nn.CrossEntropyLoss())
        dataloader (iterator): Mini-batch data iterator. Requires self.epoch_size and self.num_iters variable.
        rank (int): Rank of the process in distributed training.
        world_size (int): Total number of processes in distributed training.
        scheduler (torch scheduler): Only needed when the scheduler is ReduceLROnPlateau, and using one clip eval.
        PAD_VALUE (int): The value to be padded when each process has different number of batch size. These padded value will be removed anyway after all gathering.
    
    Returns:
        tuple: sample_seen, total_samples, loss, acc, vid_acc_top1, vid_acc_top5, elapsed_time, video_metrics (VideoMetrics), eval_log_str
    """

    cur_device = torch.cuda.current_device()

    is_scheduler_plateau = one_clip and isinstance(scheduler, ReduceLROnPlateau)

    with torch.no_grad():#{{{
        model.eval()

        # These two variables are needed for every process for ReduceLROnPlateau scheduling.
        sample_seen = 0
        loss_accum = 0.

        if rank == 0:
            num_correct_preds = 0
            total_samples = len(dataloader.dataset)
            total_iters = len(dataloader)

            video_metrics = VideoMetrics()
            #video_metrics.clean_data()     # new validation
            start_time = time.time()

            if one_clip:
                eval_mode = "One-clip Eval"
            else:
                eval_mode = "Multi-clip Eval"
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
            shard_size, num_iters, last_batch_size = count_true_samples(dataloader.sampler, batch_size)
        else:
            shard_size, num_iters, last_batch_size = total_samples, total_iters, total_samples % batch_size

        if rank == 0 :
            assert num_iters == total_iters, "Implementation error"

        #for it in range(max_num_iters):
        for it, data in enumerate(dataloader):
            inputs, uids, labels, _ = data_unpack_func(data)#{{{
            inputs, uids, labels, curr_batch_size = misc.data_to_gpu(inputs, uids, labels)

            perform_forward = it < num_iters - 1 or last_batch_size > 0     # not last batch or last batch size is at least 1

            if it == num_iters - 1:
                """last batch true data sampling"""

                
                #inputs = [inputs[0][:last_batch_size]]
                inputs = inputs[:last_batch_size]
                labels = labels[:last_batch_size]
                uids = uids[:last_batch_size]
                curr_batch_size = torch.LongTensor([last_batch_size]).to(cur_device, non_blocking=True)

#                uids = torch.IntTensor([]).to(cur_device, non_blocking=True)
#                labels = torch.IntTensor([]).to(cur_device, non_blocking=True)

            if perform_forward:
                """Not the last batch"""
                # forward
                if input_reshape_func:
                    inputs = input_reshape_func(inputs)
                outputs = model(inputs)
                batch_loss = criterion(outputs, labels)
                batch_loss_accum = batch_loss * curr_batch_size

                _, predicted = torch.max(outputs.data, 1)
                batch_correct = (predicted == labels).sum()

            else:
                #curr_batch_size = torch.LongTensor([0]).to(cur_device, non_blocking=True)
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
                #eta = int((total_samples-sample_seen) * elapsed_time / sample_seen)
                eta = int((total_iters-(it+1)) * elapsed_time / (it+1))

                if one_clip:
                    eval_mode = "One-clip Eval"
                else:
                    eval_mode = "Multi-clip Eval"
                sys.stdout.write("\r {:s} Iter: {:4d}/{:4d} - Sample: {:6d}/{:6d} - ETA: {:4d}s - val_loss: {:.4f} - val_acc: {:.4f}        ".format(eval_mode, it+1, total_iters, sample_seen, total_samples, eta, loss, acc))
                sys.stdout.flush()

        # Reset the iterator. Needs to be done at the end of epoch when __next__ is directly called instead of doing iteration.
#        dataloader.reset()

    if is_scheduler_plateau:
        with OutputLogger(scheduler.__module__, "INFO"):   # redirect stdout print() to logging (for verbose=True)
            scheduler.step(loss)

    if rank == 0:
        vid_acc_top1, vid_acc_top5 = video_metrics.accuracy(topk=(1,5))

        sys.stdout.write("\r")
        sys.stdout.flush()

        eval_log_str = " {:s} Iter: {:4d}/{:4d} - Sample: {:6d}/{:6d} - {:d}s - val_loss: {:.4f} - val_acc: {:.4f}".format(eval_mode, it+1, total_iters, sample_seen, total_samples, round(elapsed_time), loss, acc)
        if not one_clip:
            eval_log_str += " - val_vid_acc_top1: {:.4f} - val_vid_acc_top5: {:.4f}".format(vid_acc_top1, vid_acc_top5)
        logger.info(eval_log_str)

    else:
        sample_seen = None
        total_samples = None
        loss = None
        acc = None
        vid_acc_top1 = None
        vid_acc_top5 = None
        elapsed_time = None
        video_metrics = None
        eval_log_str = None

    return sample_seen, total_samples, loss, acc, vid_acc_top1, vid_acc_top5, elapsed_time, video_metrics, eval_log_str#}}}

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
            scheduler.step(loss)

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
