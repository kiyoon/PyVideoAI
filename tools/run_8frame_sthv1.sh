exp_root="$HOME/experiments_tcswap"
dataset="something_v1"
model="trn_resnet50"
telegramidx=1
#exp_name="crop224_lr001_8frame_largejit_plateau_5scrop"
#
#CUDA_VISIBLE_DEVICES=0,1 ./run_train.py --local_world_size 2 -R $exp_root -D $dataset -M $model -E $exp_name -c:e tcswap -e 200 -B $telegramidx --init_method tcp://localhost:19988
#CUDA_VISIBLE_DEVICES=0,1 ./run_val.py --local_world_size 2 -R $exp_root -D $dataset -M $model -E $exp_name -c:e tcswap -l -2 -m multicrop -B $telegramidx --init_method tcp://localhost:19988
#
#exp_name="crop224_lr001_8frame_largejit_plateau_TCswap_5scrop"
#CUDA_VISIBLE_DEVICES=0,1 ./run_train.py --local_world_size 2 -R $exp_root -D $dataset -M $model -E $exp_name -c:e tcswap -e 200 -B $telegramidx --init_method tcp://localhost:19988
#CUDA_VISIBLE_DEVICES=0,1 ./run_val.py --local_world_size 2 -R $exp_root -D $dataset -M $model -E $exp_name -c:e tcswap -l -2 -m multicrop -B $telegramidx --init_method tcp://localhost:19988

model="tsm_avg_resnet50"
exp_name="crop224_lr001_8frame_largejit_plateau_5scrop"
CUDA_VISIBLE_DEVICES=0,1 ./run_train.py --local_world_size 2 -R $exp_root -D $dataset -M $model -E $exp_name -c:e tcswap -e 200 -B $telegramidx --init_method tcp://localhost:19988 -l -1
CUDA_VISIBLE_DEVICES=0,1 ./run_val.py --local_world_size 2 -R $exp_root -D $dataset -M $model -E $exp_name -c:e tcswap -l -2 -m multicrop -B $telegramidx --init_method tcp://localhost:19988

exp_name="crop224_lr001_8frame_largejit_plateau_TCswap_5scrop"
CUDA_VISIBLE_DEVICES=0,1 ./run_train.py --local_world_size 2 -R $exp_root -D $dataset -M $model -E $exp_name -c:e tcswap -e 200 -B $telegramidx --init_method tcp://localhost:19988
CUDA_VISIBLE_DEVICES=0,1 ./run_val.py --local_world_size 2 -R $exp_root -D $dataset -M $model -E $exp_name -c:e tcswap -l -2 -m multicrop -B $telegramidx --init_method tcp://localhost:19988
