
def get_last_batch_size_singleGPU(num_samples, batch_size):
    # ONLY for single process (single GPU)
    # The logic is as follows
    #
    # if num_samples == 0:  return 0
    # elif num_samples % batch_size == 0:   return batch_size
    # else: return num_samples % batch_size
    return (num_samples-1) % batch_size + 1 if num_samples > 0 else 0

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
    # same as ceil(shard_size_padded / batch_size_per_proc) but without floating-point operations
    num_iters = shard_size_padded // batch_size_per_proc + int(shard_size_padded % batch_size_per_proc > 0)   

    # NOTE: last_batch_size can be 0.
    # Previous bug is that it is calculated by shard_size % batch_size_per_proc, but if every process shard_size is divisible, than you get last iteration of all zeros.
    last_batch_size = shard_size - (num_iters-1)*batch_size_per_proc

    return shard_size, num_iters, last_batch_size
