#!/bin/bash

function help_print {
	echo "Usage: $0 [train/eval] [NUM_GPUS_PER_NODE] [NUM_NODES] [JOB_UUID] [MASTER_ADDR[:<PORT>]] [train/eval_dist.py args...]"
	exit 1
}

if [ $# -lt 5 ]
then
	help_print
fi

MODE="$1"

if ! [[ $MODE =~ ^(train|eval)$ ]]
then
	echo "Only train or eval mode is accepted but got $MODE."
	help_print
fi

NUM_GPUS="$2"
NUM_NODES="$3"
JOB_UUID="$4"
MASTER_ADDR="$5"

# Any arguments from the 6th one are captured by ${@:6}

if [ $MODE == 'train' ]
then
	python -m torch.distributed.run 	\
		--nnodes="$NUM_NODES"			\
		--nproc_per_node="$NUM_GPUS"	\
		--rdzv_id="$JOB_UUID"			\
		--rdzv_backend=c10d				\
		--rdzv_endpoint="$MASTER_ADDR"	\
		$(dirname "$0")/train_dist.py ${@:6}
else
	python -m torch.distributed.run 	\
		--nnodes="$NUM_NODES"			\
		--nproc_per_node="$NUM_GPUS"	\
		--rdzv_id="$JOB_UUID"			\
		--rdzv_backend=c10d				\
		--rdzv_endpoint="$MASTER_ADDR"	\
		$(dirname "$0")/eval_dist.py ${@:6}
fi
