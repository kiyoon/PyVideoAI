#!/bin/bash

function help_print {
	echo "Usage: $0 [train/eval/feature] [NUM_GPUS] [train/eval_dist.py args...]"
	exit 1
}

if [ $# -lt 2 ]
then
	help_print
fi

MODE="$1"

if ! [[ $MODE =~ ^(train|eval|feature)$ ]]
then
	echo "Only train, eval and feature modes are accepted but got $MODE."
	help_print
fi

NUM_GPUS="$2"
# Any arguments from the third one are captured by ${@:3}

if [[ $MODE == 'train' ]]
then
	if [[ $NUM_GPUS -eq 1 ]]
	then
		python $(dirname "$0")/run_train.py ${@:3}
	else
		python -m torch.distributed.run 	\
			--standalone 					\
			--nnodes=1						\
			--nproc_per_node="$NUM_GPUS"	\
			$(dirname "$0")/train_dist.py ${@:3}
	fi
elif [[ $MODE == 'eval' ]]
then
	if [[ $NUM_GPUS -eq 1 ]]
	then
		python $(dirname "$0")/run_eval.py ${@:3}
	else
		python -m torch.distributed.run 	\
			--standalone 					\
			--nnodes=1						\
			--nproc_per_node="$NUM_GPUS"	\
			$(dirname "$0")/eval_dist.py ${@:3}
	fi
else
	if [[ $NUM_GPUS -eq 1 ]]
	then
		python $(dirname "$0")/run_feature_extract.py ${@:3}
	else
		python -m torch.distributed.run 	\
			--standalone 					\
			--nnodes=1						\
			--nproc_per_node="$NUM_GPUS"	\
			$(dirname "$0")/feature_extract_dist.py ${@:3}
	fi
fi
