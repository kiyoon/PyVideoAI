
def count_true_samples(distributed_sampler, batch_size_per_proc):
    """

    For example:
    If the world size is 8 and the epoch size (total number of samples) is 10,
    Each process will return batch size of
    2 2 2 2 2 2 2 -> with padding (default)
    2 2 1 1 1 1 1 -> real samples

    This function returns the real shard size, the number of iterations, and the last batch size. You just have to resample the last batch when it's the end of the epoch.
    Note the edge case that some last batch size can be 0.
    """

    world_size = distributed_sampler.num_replicas
    rank = distributed_sampler.rank
    total_samples = len(distributed_sampler.dataset)
    shard_size_padded = distributed_sampler.num_samples

    # Real number of samples for this process. Different across ranks.
    shard_size = total_samples // world_size + int(rank < total_samples % world_size)

    # Because of padding, num_iters is the same across all processes.
    num_iters = shard_size_padded // batch_size_per_proc + int(shard_size_padded % batch_size_per_proc > 0)   

    last_batch_size = shard_size % batch_size_per_proc

    return shard_size, num_iters, last_batch_size
