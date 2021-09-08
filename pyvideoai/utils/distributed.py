#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Distributed helpers."""

import functools
import logging
import pickle
import torch
import torch.distributed as dist
import os
import contextlib
import sys

import numpy as np
import random
import json

logger = logging.getLogger(__name__)

def all_gather(tensors):
    """
    All gathers the provided tensors from all processes across machines.
    Args:
        tensors (list): tensors to perform all gather across all processes in
        all machines.
    """

    gather_list = []
    output_tensor = []
    world_size = dist.get_world_size()
    for tensor in tensors:
        tensor_placeholder = [
            torch.ones_like(tensor) for _ in range(world_size)
        ]
        dist.all_gather(tensor_placeholder, tensor, async_op=False)
        gather_list.append(tensor_placeholder)
    for gathered_tensor in gather_list:
        output_tensor.append(torch.cat(gathered_tensor, dim=0))
    return output_tensor


def all_reduce(tensors, average=True):
    """
    All reduce the provided tensors from all processes across machines.
    Args:
        tensors (list): tensors to perform all reduce across all processes in
        all machines.
        average (bool): scales the reduced tensor by the number of overall
        processes across all machines.
    """

    for tensor in tensors:
        dist.all_reduce(tensor, async_op=False)
    if average:
        world_size = dist.get_world_size()
        for tensor in tensors:
            tensor.mul_(1.0 / world_size)
    return tensors


def init_process_group(
    local_rank,
    local_world_size,
    shard_id,
    num_shards,
    init_method,
    dist_backend="nccl",
):
    """
    Initializes the default process group.
    Args:
        local_rank (int): the rank on the current local machine.
        local_world_size (int): the world size (number of processes running) on
        the current local machine.
        shard_id (int): the shard index (machine rank) of the current machine.
        num_shards (int): number of shards for distributed training.
        init_method (string): supporting three different methods for
            initializing process groups:
            "file": use shared file system to initialize the groups across
            different processes.
            "tcp": use tcp address to initialize the groups across different
        dist_backend (string): backend to use for distributed training. Options
            includes gloo, mpi and nccl, the details can be found here:
            https://pytorch.org/docs/stable/distributed.html
    """
    # Sets the GPU to use.
    torch.cuda.set_device(local_rank)
    # Initialize the process group.
    proc_rank = local_rank + shard_id * local_world_size
    world_size = local_world_size * num_shards
    dist.init_process_group(
        backend=dist_backend,
        init_method=init_method,
        world_size=world_size,
        rank=proc_rank,
    )


def is_master_proc(num_gpus=8):
    """
    Determines if the current process is the master process.
    """
    if torch.distributed.is_initialized():
        return dist.get_rank() % num_gpus == 0
    else:
        return True


def get_world_size():
    """
    Get the size of the world.
    """
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """
    Get the rank of the current process.
    """
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier(device_ids=[torch.cuda.current_device()])


@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    Returns:
        (group): pytorch dist group.
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD


def _serialize_to_tensor(data, group):
    """
    Seriialize the tensor to ByteTensor. Note that only `gloo` and `nccl`
        backend is supported.
    Args:
        data (data): data to be serialized.
        group (group): pytorch dist group.
    Returns:
        tensor (ByteTensor): tensor that serialized.
    """

    backend = dist.get_backend(group)
    assert backend in ["gloo", "nccl"]
    device = torch.device("cpu" if backend == "gloo" else "cuda")

    buffer = pickle.dumps(data)
    if len(buffer) > 1024 ** 3:
        logger = logging.getLogger(__name__)
        logger.warning(
            "Rank {} trying to all-gather {:.2f} GB of data on device {}".format(
                get_rank(), len(buffer) / (1024 ** 3), device
            )
        )
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)
    return tensor


def _pad_to_largest_tensor(tensor, group):
    """
    Padding all the tensors from different GPUs to the largest ones.
    Args:
        tensor (tensor): tensor to pad.
        group (group): pytorch dist group.
    Returns:
        list[int]: size of the tensor, on each rank
        Tensor: padded tensor that has the max size
    """
    world_size = dist.get_world_size(group=group)
    assert (
        world_size >= 1
    ), "comm.gather/all_gather must be called from ranks within the given group!"
    local_size = torch.tensor(
        [tensor.numel()], dtype=torch.int64, device=tensor.device
    )
    size_list = [
        torch.zeros([1], dtype=torch.int64, device=tensor.device)
        for _ in range(world_size)
    ]
    dist.all_gather(size_list, local_size, group=group)
    size_list = [int(size.item()) for size in size_list]

    max_size = max(size_list)

    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    if local_size != max_size:
        padding = torch.zeros(
            (max_size - local_size,), dtype=torch.uint8, device=tensor.device
        )
        tensor = torch.cat((tensor, padding), dim=0)
    return size_list, tensor


def all_gather_unaligned(data, group=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.

    Returns:
        list[data]: list of data gathered from each rank
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    if dist.get_world_size(group) == 1:
        return [data]

    tensor = _serialize_to_tensor(data, group)

    size_list, tensor = _pad_to_largest_tensor(tensor, group)
    max_size = max(size_list)

    # receiving Tensor from all ranks
    tensor_list = [
        torch.empty((max_size,), dtype=torch.uint8, device=tensor.device)
        for _ in size_list
    ]
    dist.all_gather(tensor_list, tensor, group=group)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def write_pids_to_file(path_to_file):
    """
    Write pids to the file to make killing the processes easier.
    """

    cur_device = torch.cuda.current_device()

    proc_ids = torch.LongTensor([os.getpid()]).to(cur_device)
    (proc_ids,) = all_gather([proc_ids])

    if get_rank() == 0:
        with open(path_to_file, "w") as f:
            for proc_id in proc_ids:
                f.write("%d\n" % proc_id)


def check_random_seed_in_sync(num_values_per_process=3):
    """
    Check if random seed is in sync across processes.
    This is REQUIRED especially for model initialisation, because different seed will make parameters different across processes (GPUs).
    This will be desynced when: (AVOID below)
        1. Each process has different random seed
        2. Each process has different number of random calls.
    """
    if get_world_size() > 1:
        cur_device = torch.cuda.current_device()

        # PyTorch
        rand_values = torch.rand(1,num_values_per_process).to(cur_device)
        (rand_values,) = all_gather([rand_values])          # shape: (num_processes, num_values_per_process)
        if rand_values.unique(dim=0).size(0) != 1:          # shape has to be (1, num_values_per_process) if all values are equal over the processes.
            raise RuntimeError('PyTorch random seed not in sync over the multiple processes. Make sure you are not calling more random calls on only some of the processes.')

        # numpy
        rand_values = torch.from_numpy(np.random.rand(1,num_values_per_process)).to(cur_device)
        (rand_values,) = all_gather([rand_values])          # shape: (num_processes, num_values_per_process)
        if rand_values.unique(dim=0).size(0) != 1:          # shape has to be (1, num_values_per_process) if all values are equal over the processes.
            raise RuntimeError('Numpy random seed not in sync over the multiple processes. Make sure you are not calling more random calls on only some of the processes.')

        # random 
        rand_array = np.zeros((1, num_values_per_process))
        for i in range(num_values_per_process):
            rand_array[0, i] = random.random()

        rand_values = torch.from_numpy(np.random.rand(1,num_values_per_process)).to(cur_device)
        (rand_values,) = all_gather([rand_values])          # shape: (num_processes, num_values_per_process)
        if rand_values.unique(dim=0).size(0) != 1:          # shape has to be (1, num_values_per_process) if all values are equal over the processes.
            raise RuntimeError('Python-random random seed not in sync over the multiple processes. Make sure you are not calling more random calls on only some of the processes.')

class MultiprocessPrinter:
    '''In every print, show which process rank is printing the message

    with MultiprocessPrinter(output_stream=sys.__stdout__):
        print('debug message')

    This will print:
    rank 0: debug message
    rank 1: debug message
    ...

    '''
    def __init__(self, rank=None, output_stream=sys.stdout):
        if rank is None:
            self.rank = get_rank()
        else:
            self.rank = rank
        self._redirector = contextlib.redirect_stdout(self)
        self.output_stream = output_stream

    def write(self, msg):
        if msg and not msg.isspace():
            self.output_stream.write(f'rank {self.rank}: {msg}\n')


    def flush(self):
        self.output_stream.flush()

    def __enter__(self):
        self._redirector.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # let contextlib do any exception handling here
        self._redirector.__exit__(exc_type, exc_value, traceback)


def suppress_print():
    """
    Suppresses printing from the current process.
    """

    sys.stdout = open(os.devnull,'w')
    sys.stderr = open(os.devnull,'w')


def distributed_init(global_seed, local_world_size = None):
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if local_world_size:
        # DDP spawn
        local_rank = rank % local_world_size
    else:
        # torch.distributed.run
        local_world_size = int(os.getenv('LOCAL_WORLD_SIZE', 1)) if dist.is_initialized() else 1
        local_rank = int(os.getenv('LOCAL_RANK', 0)) if dist.is_initialized() else 0


    """You have to set the seed equally across processes.
    Reason: during model initialisation and training, the models across processes must be in sync.
    On the other hand, dataloader will have multiple workers with different seed, so dataloading randomness will be different across processes.
    """
    #local_seed = global_seed + rank
    local_seed = global_seed

    # Set GPU
    torch.cuda.set_device(local_rank)

    # Set seed
    torch.manual_seed(local_seed)
    torch.cuda.manual_seed(local_seed)
    np.random.seed(local_seed)
    random.seed(local_seed)
    # DALI seed

    return rank, world_size, local_rank, local_world_size, local_seed


def log_distributed_info(world_size, local_world_size):
    distributed_configs = {'world_size': world_size,
            'local_world_size': local_world_size,
            'num_nodes (estimated)': world_size // local_world_size}
    logger.info("distributed configs: " + json.dumps(distributed_configs, sort_keys=False, indent=4))


def seed_worker(worker_id):
    """By default, each worker will have its PyTorch seed set to base_seed + worker_id.
    However, seeds for other libraries may be duplicated upon initializing workers, causing each worker to return identical random numbers.
    This function seeds numpy and random's random seed.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

