# Capturing Temporal Information in a Single Frame: Channel Sampling Strategies for Action Recognition

Kiyoon Kim, Shreyank N Gowda, Oisin Mac Aodha, Laura Sevilla-Lara  
In BMVC 2022. [`arXiv`](http://arxiv.org/abs/2201.10394)

<img src="https://user-images.githubusercontent.com/12980409/151038213-12bdad91-7895-40e7-9304-126079fed637.png" alt="8-frame TC Reordering" width="800">  
<img src="https://user-images.githubusercontent.com/12980409/151038200-6f32cea8-6a2b-40bf-9d95-50ba860114be.png" alt="3-frame GrayST" width="800">  

## Getting started

### PyVideoAI checklist
- Install PyVideoAI correctly. See [README.md](../../../README.md) and [INSTALL.md](../../../docs/INSTALL.md).
- Following [PyVideoAI-examples](https://github.com/kiyoon/PyVideoAI-examples) help understand the overall workflow.

### Preparing the datasets
#### Something-Something-V1
1. Download the dataset and annotations. Rename the directories into `frames` and `annotations`, and put them in `data/something-something-v1`.
2. Generate splits.

```bash
conda activate videoai
python tools/datasets/generate_somethingv1_splits.py data/something-something-v1/splits_frames data/something-something-v1/annotations --root data/something-something-v1/frames --mode frames
```

#### Something-Something-V2
1. Download the dataset and annotations. Rename the directories into `videos` and `annotations`, and put them in `data/something-something-v2`.
2. Extract videos into frames of images, to folder `data/something-something-v2/frames_q5`.

```bash
submodules/video_datasets_api/tools/something-something-v2/extract_frames.sh data/something-something-v2/videos data/something-something-v2/frames_q5
```

3. Generate splits.

```bash
conda activate videoai
python tools/datasets/generate_somethingv2_splits.py data/something-something-v2/splits_frames data/something-something-v2/annotations data/something-something-v2/frames_q5 --mode frames
```

### About the paper
- Core implementation of reordering methods is in [`pyvideoai/utils/tc_reordering.py`](../../../pyvideoai/utils/tc_reordering.py).  
- See [`exp_configs/ch_tcgrey`](../../../exp_configs/ch_tcgrey) for experiment settings.  
For example, in order to run TSM model, GreyST method on the Something-Something-V1 dataset, you should run [`exp_configs/ch_tcgrey/something_v1/tsm_resnet50_nopartialbn-GreyST_8frame.py`](../../../exp_configs/ch_tcgrey/something_v1/tsm_resnet50_nopartialbn-GreyST_8frame.py), the command of which would be:

```bash
# Run training
tools/run_singlenode.sh train 4 -R ~/experiment_root -D something_v1 -M tsm_resnet_nopartialbn -E GreyST_8frame -c:e tcgrey
# Run evaluation
tools/run_singlenode.sh eval 4 -R ~/experiment_root -D something_v1 -M tsm_resnet_nopartialbn -E GreyST_8frame -c:e tcgrey
```

#### Once you prepared the datasets, just modify the script below and run.  
```bash
#!/bin/bash
exp_root="$HOME/experiments"  # Experiment results will be saved here.

export CUDA_VISIBLE_DEVICES=0
num_gpus=1

subfolder="test_run"           # Name subfolder as you like.

## Choose the dataset
dataset=something_v1
#dataset=something_v2
#dataset=cater_task2
#dataset=cater_task2_cameramotion

## Choose the model
model=tsn_resnet50
#model=trn_resnet50
#model=mtrn_resnet50
#model=tsm_resnet50_nopartialbn     # NOTE: use tsm_resnet50 for CATER experiments
#model=mvf_resnet50_nopartialbn     # NOTE: use mvf_resnet50 for CATER experiments

## Choose the sampling method.
## NOTE: Use 32 frame for CATER experiments.
exp_name="RGB_8frame"
#exp_name="TC_8frame"
#exp_name="TCPlus2_8frame"
#exp_name="GreyST_8frame"

# Training script
# -S creates a subdirectory in the name of your choice. (optional)
tools/run_singlenode.sh train $num_gpus -R $exp_root -D $dataset -M $model -E $exp_name -c:e tcgrey -S "$subfolder" #--wandb_project kiyoon_kim_tcgrey

# Evaluating script
# -l -2 loads the best model
# -p saves the predictions. (optional)
tools/run_singlenode.sh eval $num_gpus -R $exp_root -D $dataset -M $model -E $exp_name -c:e tcgrey -S "$subfolder" -l -2 -p #--wandb
```

## Citing the paper

If you find our work or code useful, please cite:

```BibTeX
@inproceedings{kim2022capturing,
  title={Capturing Temporal Information in a Single Frame: Channel Sampling Strategies for Action Recognition},
  author={Kim, Kiyoon and Gowda, Shreyank N and Mac Aodha, Oisin and Sevilla-Lara, Laura},
  booktitle={British Machine Vision Conference (BMVC)},
  year={2022}
}
