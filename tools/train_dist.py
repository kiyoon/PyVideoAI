#!/usr/bin/env python3

"""Train using multi-GPU, single or multi-node.

Use `torch.distributed.run` (PyTorch >= 1.9.0) or `torch.distributed.launch` (PyTorch <= 1.8.1) to run this code.

Usage
~~~~~
1. Single-node multi-GPU
::
    >>> python -m torch.distributed.run
        --standalone
        --nnodes=1
        --nproc_per_node=$NUM_GPUS
        train_dist.py (--arg1 ... train script args...)
2. Multi-node multi-GPU:
::
    >>> python -m torch.distributed.run
        --nnodes=$NUM_NODES
        --nproc_per_node=$NUM_GPUS
        --rdzv_id=$JOB_ID
        --rdzv_backend=c10d
        --rdzv_endpoint=$HOST_NODE_ADDR
        train_dist.py (--arg1 ... train script args...)
``HOST_NODE_ADDR``, in form <host>[:<port>] (e.g. node1.example.com:29400), specifies the node and
the port on which the C10d rendezvous backend should be instantiated and hosted. It can be any
node in your training cluster, but ideally you should pick a node that has a high bandwidth.
.. note::
   If no port number is specified ``HOST_NODE_ADDR`` defaults to 29400.


"""

import argparse
import torch

from pyvideoai.train_multiprocess import train

from run_train import add_train_args

def get_parser():
    parser = argparse.ArgumentParser(description="Train an action model",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_train_args(parser)
    return parser


def main():
    """
    Main function to spawn the train and test process.
    """
    parser = get_parser()
    args = parser.parse_args()
    torch.distributed.init_process_group(backend='nccl')
    # torch.cuda.set_device not needed?

    train(args)



if __name__ == "__main__":
    main()
