# An Action Is Worth Multiple Words: Handling Ambiguity in Action Recognition

Kiyoon Kim, Davide Moltisanti, Oisin Mac Aodha, Laura Sevilla-Lara  
In BMVC 2022. [`arXiv`](https://arxiv.org/abs/2210.04933)

<img src="https://user-images.githubusercontent.com/12980409/193856345-e0287624-4c84-46af-b245-c07ff263c424.png" alt="Verb Ambiguity" width="800"> 

<img src="https://user-images.githubusercontent.com/12980409/193883304-9e7275ee-2b88-4250-b695-c6d2f0d5acc0.png" alt="Method" width="1000"> 

## Dataset downloads (labels only)
- [EPIC-Kitchens-100-SPMV test set labels](https://github.com/kiyoon/verb_ambiguity/releases/download/datasets-v1.0.0/ek100-val-multiple-verbs-halfagree-halfconfident-include_original-20220427.csv)
- [Confusing-HMDB-102 labels](https://github.com/kiyoon/verb_ambiguity/releases/download/datasets-v1.0.0/confusing_hmdb_102_splits.tar.gz)

## Running feature experiments using pre-extracted features
1. Download pre-extracted features.
- [Download EPIC-Kitchens-100 TSM features](https://github.com/kiyoon/verb_ambiguity/releases/download/datasets-v1.0.0/EPIC_KITCHENS_100_TSM_features.tar.gz)
- [Download EPIC-Kitchens-100 TSM feature neighbours (optional)](https://github.com/kiyoon/verb_ambiguity/releases/download/datasets-v1.0.0/EPIC_KITCHENS_100_TSM_neighbour_cache.tar.gz): Using this neighbour cache will reduce the preparation time of the training by skipping neighbour search.
- [Download Confusing-HMDB-102 TSM features](https://github.com/kiyoon/verb_ambiguity/releases/download/datasets-v1.0.0/confusing_hmdb_102_TSM_features.tar.gz)

2. Exract in `data/EPIC_KITCHENS_100` or `data/hmdb51`.
3. Run the training code. Change the dataset and exp_name variables to select different experiments.
 
```bash
#!/bin/bash
exp_root="$HOME/experiments"  # Experiment results will be saved here.

export CUDA_VISIBLE_DEVICES=0
num_gpus=1
export VAI_USE_NEIGHBOUR_CACHE=True     # Only for EPIC-Kitchens-100-SPMV. It will bypass neighbour search if the cache is available, otherwise it will run and cache the results.
export VAI_NUM_NEIGHBOURS=15
export VAI_PSEUDOLABEL_THR=0.1

subfolder="k=$VAI_NUM_NEIGHBOURS,thr=$VAI_PSEUDOLABEL_THR"           # Name subfolder as you like.

dataset=epic100_verb_features
#dataset=confusing_hmdb_102_features

exp_name="concat_RGB_flow_assume_negative"
#exp_name="concat_RGB_flow_weak_assume_negative"
#exp_name="concat_RGB_flow_binary_labelsmooth"
#exp_name="concat_RGB_flow_binary_negative_labelsmooth"
#exp_name="concat_RGB_flow_binary_focal"
#exp_name="concat_RGB_flow_entropy_maximise"
#exp_name="concat_RGB_flow_mask_binary_ce"
#exp_name="concat_RGB_flow_pseudo_single_binary_ce"

# Training script
# -S creates a subdirectory in the name of your choice. (optional)
tools/run_singlenode.sh train $num_gpus -R $exp_root -D $dataset -c:d verbambig -M ch_beta.featuremodel -E $exp_name -c:e verbambig -S "$subfolder" #--wandb_project kiyoon_kim_verbambig

# Evaluating script
# -l -2 loads the best model (with the highest heldout validation accuracy)
# -p saves the predictions. (optional)
tools/run_singlenode.sh eval $num_gpus -R $exp_root -D $dataset -c:d verbambig -M ch_beta.featuremodel -E $exp_name -c:e verbambig -S "$subfolder" -l -2 -p #--wandb
```

## Running feature extraction or end-to-end experiments.

### Prepare the dataset
#### EPIC-Kitchens-100-SPMV
1. Download `rgb_frames` and `flow_frames`. [`script`](https://github.com/epic-kitchens/epic-kitchens-download-scripts).  
  Extract tar files. [`RGB script`](https://github.com/kiyoon/video_datasets_api/blob/master/tools/epic_kitchens_100/epic_extract_rgb.sh), [`flow script`](https://github.com/kiyoon/video_datasets_api/blob/master/tools/epic_kitchens_100/epic_extract_flow.sh).
2. Clone [EPIC-Kitchens-100 annotations](https://github.com/epic-kitchens/epic-kitchens-100-annotations) at `data/EPIC_KITCHENS_100/epic-kitchens-100-annotations`.
3. Gulp the dataset. First, generate flow annotations using [this](https://github.com/kiyoon/video_datasets_api/blob/master/tools/epic_kitchens_100/epic_convert_rgb_to_flow_frame_idxs.py) and use [this](https://github.com/kiyoon/video_datasets_api/blob/master/tools/epic_kitchens_100/gulp_data.py) to gulp.
4. Generate dataset split files. [`RGB_script`](../../../tools/datasets/generate_epic100_splits.py), [`flow_script`](../../../tools/datasets/generate_epic100_flow_splits.py)
5. Get TSM pre-trained models from [EPIC-Kitchens Action Models](https://github.com/epic-kitchens/C1-Action-Recognition-TSN-TRN-TSM#pretrained-models), and save them into `data/pretrained/epic100`.
6. Download the [multi-verb annotations](https://github.com/kiyoon/verb_ambiguity/releases/download/datasets-v1.0.0/ek100-val-multiple-verbs-halfagree-halfconfident-include_original-20220427.csv) at `data/EPIC_KITCHENS_100/ek100-val-multiple-verbs-halfagree-halfconfident-include_original-20220427.csv`.
6. `data/EPIC_KITCHENS_100` directory should have five directories and one file: `epic-kitchens-100-annotations`, `splits_gulp_flow`, `splits_gulp_rgb`, `gulp_flow`, `gulp_rgb`, `ek100-val-multiple-verbs-halfagree-halfconfident-include_original-20220427.csv`.

#### Confusing-HMDB-102
1. Download HMDB-51 videos. [`script`](https://github.com/kiyoon/video_datasets_api/blob/master/tools/hmdb/download_hmdb.sh)
2. Extract them into frames of images. [`script`](https://github.com/kiyoon/video_datasets_api/blob/master/tools/hmdb/hmdb_extract_frames.sh)
3. Generate optical flow. [`script`](https://github.com/kiyoon/video_datasets_api/blob/master/tools/hmdb/extract_flow_multigpu.sh)
4. Gulp the dataset. [`script`](https://github.com/kiyoon/video_datasets_api/blob/master/tools/gulp_jpeg_dir.py) (Use `rgb` and `flow_onefolder` modality, and `--class_folder`).
5. Generate dataset split files. [`script`](../../../tools/datasets/generate_hmdb_splits.py) (Use `--confusion 2`) Or just download the splits.
6. `data/hmdb51` directory must have at least four directories: `confusing102_splits_gulp_flow`, `confusing102_splits_gulp_rgb`, `gulp_flow`, `gulp_rgb`.

Putting all together,
```bash
# Install unrar, nvidia-docker
# Execute from the root directory of this repo.
# Don't run all of them together. Some things may not run 

GPU_arch=pascal  # pascal / turing / ampere

conda activate videoai
submodules/video_datasets_api/tools/hmdb/download_hmdb.sh data/hmdb51
submodules/video_datasets_api/tools/hmdb/hmdb_extract_frames.sh data/hmdb51/videos data/hmdb51/frames
submodules/video_datasets_api/tools/hmdb/extract_flow_multigpu.sh data/hmdb51/frames data/hmdb51/flow $GPU_arch 0
python submodules/video_datasets_api/tools/gulp_jpeg_dir.py data/hmdb51/frames data/hmdb51/gulp_rgb rgb --class_folder
python submodules/video_datasets_api/tools/gulp_jpeg_dir.py data/hmdb51/flow data/hmdb51/gulp_flow flow_onefolder --class_folder
python tools/datasets/generate_hmdb_splits.py data/hmdb51/gulp_rgb data/hmdb51/confusing102_splits_gulp_rgb data/hmdb51/testTrainM
ulti_7030_splits --mode gulp --confusion 2
python tools/datasets/generate_hmdb_splits.py data/hmdb51/gulp_rgb data/hmdb51/confusing102_splits_gulp_rgb data/hmdb51/testTrainM
ulti_7030_splits --mode gulp --confusion 2
```


### Run training, evaluation and feature extraction.

```bash
#!/bin/bash

exp_root="$HOME/experiments"  # Experiment results will be saved here.

export CUDA_VISIBLE_DEVICES=0
num_gpus=1
export VAI_NUM_NEIGHBOURS=15
export VAI_PSEUDOLABEL_THR=0.1


## Choose dataset
#dataset=epic100_verb
dataset=confusing_hmdb_102
export VAI_SPLITNUM=1   # only for confusing_hmdb_102 dataset.

## Choose model (RGB or flow)
model="tsm_resnet50_nopartialbn"
#model="ch_epic100.tsm_resnet50_flow"

## Choose loss
## For feature extraction, use "ce"
exp_name="ce"
#exp_name="assume_negative"
#exp_name="weak_assume_negative"
#exp_name="binary_labelsmooth"
#exp_name="binary_negative_labelsmooth"
#exp_name="binary_focal"
#exp_name="entropy_maximise"
#exp_name="mask_binary_ce"
#exp_name="pseudo_single_binary_ce"


# Name subfolder as you like.
if [[ $dataset == "epic100_verb" ]]
then
    subfolder="k=$VAI_NUM_NEIGHBOURS,thr=$VAI_PSEUDOLABEL_THR"
    extra_args=()
else
    subfolder="k=$VAI_NUM_NEIGHBOURS,thr=$VAI_PSEUDOLABEL_THR,split=$VAI_SPLITNUM"
    extra_args=(-c:d verbambig)
fi

# Training script
# -S creates a subdirectory in the name of your choice. (optional)
tools/run_singlenode.sh train $num_gpus -R $exp_root -D $dataset -M $model -E $exp_name -c:e verbambig -S "$subfolder" ${extra_args[@]} #--wandb_project kiyoon_kim_verbambig

if [[ $dataset == "epic100_verb" ]]
then
# Evaluating script
# -l -2 loads the best model (with the highest heldout validation accuracy)
# -p saves the predictions. (optional)
tools/run_singlenode.sh eval $num_gpus -R $exp_root -D $dataset -M $model -E $exp_name -c:e verbambig -S "$subfolder" -l -2 -p ${extra_args[@]} #--wandb
else
    echo "For Confusing-HMDB-102, there is no evaluation script. See summary.csv file and get the best number per metric."
fi


if [[ $exp_name == "ce" ]]
then
# Extract features
# -l -2 loads the best model (with the highest heldout validation accuracy)
tools/run_singlenode.sh feature $num_gpus -R $exp_root -D $dataset -M $model -E $exp_name -c:e verbambig -S "$subfolder" -l -2 -s traindata_testmode ${extra_args[@]} #--wandb
tools/run_singlenode.sh feature $num_gpus -R $exp_root -D $dataset -M $model -E $exp_name -c:e verbambig -S "$subfolder" -l -2 -s val ${extra_args[@]} #--wandb
fi
```

Once features are extracted, copy to `data/` directory and edit `dataset_configs/ch_verbambig/epic100_verb_features.py` or `dataset_configs/ch_verbambig/confusing_hmdb_102_features.py` to update the corresponding feature path.  

Refer to the [Running feature experiments using pre-extracted features](#running-feature-experiments-using-pre-extracted-features) section for running experiments using the features.

## Citing the paper

If you find our work or code useful, please cite:

```BibTeX
@inproceedings{kim2022ambiguity,
  title={An Action Is Worth Multiple Words: Handling Ambiguity in Action Recognition},
  author={Kim, Kiyoon and Moltisanti, Davide and Mac Aodha, Oisin and Sevilla-Lara, Laura},
  booktitle={British Machine Vision Conference (BMVC)},
  year={2022}
}
