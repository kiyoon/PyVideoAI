#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multiprocessing helpers."""

import torch


def run(
    local_rank, local_world_size, func, init_method, shard_id, num_shards, backend, args
):
    """
    Runs a function from a child process.
    Args:
        local_rank (int): rank of the current process on the current machine.
        local_world_size (int): number of processes per machine.
        func (function): function to execute on each of the process.
        init_method (string): method to initialize the distributed training.
            TCP initialization: equiring a network address reachable from all
            processes followed by the port.
            Shared file-system initialization: makes use of a file system that
            is shared and visible from all machines. The URL should start with
            file:// and contain a path to a non-existent file on a shared file
            system.
        shard_id (int): the rank of the current machine.
        num_shards (int): number of overall machines for the distributed
            training job.
        backend (string): three distributed backends ('nccl', 'gloo', 'mpi') are
            supports, each with different capabilities. Details can be found
            here:
            https://pytorch.org/docs/stable/distributed.html
    """
    # Initialize the process group.
    world_size = local_world_size * num_shards
    rank = shard_id * local_world_size + local_rank

    try:
        torch.distributed.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
        )
    except Exception as e:
        raise e

    func(args)
