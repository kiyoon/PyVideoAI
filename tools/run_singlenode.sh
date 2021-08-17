#!/bin/bash

function help_print {
	echo "Usage: $0 [train/eval] [NUM_GPUS] [train/eval_dist.py args...]"
	exit 1
}

if [ $# -lt 2 ]
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
# Any arguments from the third one are captured by ${@:3}

if [ $MODE == 'train' ]
then
	python -m torch.distributed.run 	\
		--standalone 					\
		--nnodes=1						\
		--nproc_per_node="$NUM_GPUS"	\
		$(dirname "$0")/train_dist.py ${@:3}
else
	python -m torch.distributed.run 	\
		--standalone 					\
		--nnodes=1						\
		--nproc_per_node="$NUM_GPUS"	\
		$(dirname "$0")/eval_dist.py ${@:3}
fi
