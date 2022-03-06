#!/bin/bash

function help_print {
	echo "Usage: $0 [train/eval/feature] [NUM_GPUS_PER_NODE] [NUM_NODES] [NODE_RANK] [MASTER_ADDR] [MASTER_PORT] [train/eval_dist.py args...]"
	exit 1
}

if [ $# -lt 6 ]
then
	help_print
fi

MODE="$1"

if ! [[ $MODE =~ ^(train|eval|feature)$ ]]
then
	echo "Only train, eval, or feature mode is accepted but got $MODE."
	help_print
fi

NUM_GPUS="$2"
NUM_NODES="$3"
NODE_RANK="$4"
MASTER_ADDR="$5"
MASTER_PORT="$6"

# Any arguments from the 7th one are captured by ${@:7}

if [ $MODE == 'train' ]
then
	python -m torch.distributed.launch 	\
		--use_env						\
		--nnode="$NUM_NODES"			\
		--nproc_per_node="$NUM_GPUS"	\
		--node_rank="$NODE_RANK"		\
		--master_addr="$MASTER_ADDR"	\
		--master_port="$MASTER_PORT"	\
		$(dirname "$0")/train_dist.py ${@:7}
elif [ $MODE == 'eval' ]
then
	python -m torch.distributed.launch 	\
		--use_env						\
		--nnode="$NUM_NODES"			\
		--nproc_per_node="$NUM_GPUS"	\
		--node_rank="$NODE_RANK"		\
		--master_addr="$MASTER_ADDR"	\
		--master_port="$MASTER_PORT"	\
		$(dirname "$0")/eval_dist.py ${@:7}
else
	python -m torch.distributed.launch 	\
		--use_env						\
		--nnode="$NUM_NODES"			\
		--nproc_per_node="$NUM_GPUS"	\
		--node_rank="$NODE_RANK"		\
		--master_addr="$MASTER_ADDR"	\
		--master_port="$MASTER_PORT"	\
		$(dirname "$0")/feature_extract_dist.py ${@:7}
fi
