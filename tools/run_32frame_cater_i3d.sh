exp_root="$HOME/experiments_tcswap"
dataset="cater_task2"
model="i3d_resnet50"
telegramidx=1

exp_name="crop224_32x8_largejit_plateau_3scrop10tcrop"
#CUDA_VISIBLE_DEVICES=0,1 ./run_train.py --local_world_size 2 -R $exp_root -D $dataset -M $model -E $exp_name -c:e tcswap -e 200 -B $telegramidx --init_method tcp://localhost:19988
CUDA_VISIBLE_DEVICES=0,1 ./run_eval.py --local_world_size 2 -R $exp_root -D $dataset -M $model -E $exp_name -c:e tcswap -l -2 -m multicrop -B $telegramidx --init_method tcp://localhost:19988
